"""Evaluacion del pipeline RAG de MultaClara con Ragas.

Flujo (cada paso guarda su resultado en eval/runs/ y es *reanudable*, porque
la cuota gratuita de Gemini puede cortar una corrida a la mitad):

  1. GENERAR : corre cada pregunta de eval/dataset_eval.json por el RAG real
               (recuperacion + generacion) -> eval/runs/<tag>_respuestas.json
  2. EVALUAR : calcula con Ragas faithfulness, answer_relevancy,
               context_precision y context_recall -> eval/runs/<tag>_ragas.json
  3. RESUMEN : tabla de metricas -> eval/runs/<tag>_resumen.md

Adaptacion a la salida JSON: Ragas espera texto, no un JSON. Se extrae la
justificacion en prosa del veredicto (`fundamento` + explicaciones de los
hallazgos, o `respuesta_fuera_de_alcance`) y esa es la `response` evaluada.

Ejemplos:
    python ragas_eval.py --tag baseline
    python ragas_eval.py --tag topk3 --top-k 3 --comparar baseline
    python ragas_eval.py --tag chunk800 --collection exp_800 --comparar baseline
    python ragas_eval.py --tag baseline --solo-generar        # sin gastar juez
"""

import argparse
import dataclasses
import json
import math
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from src.assistant import AssistantError, MultaClaraAssistant  # noqa: E402
from src.config import AppConfig, ConfigError  # noqa: E402
from src.schemas import Clasificacion, VeredictoMulta  # noqa: E402

DATASET = Path("eval/dataset_eval.json")
RUNS = Path("eval/runs")
METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# Frases con las que el asistente reconoce que la base no cubre el punto.
_RE_NO_CUBRE = re.compile(
    r"no\s+(?:se\s+)?(?:precis|establec|especific|contien|cubr|encuentr|determin|consta|incluy)|"
    r"no\s+(?:es\s+posible|hay\s+informaci)|fuera\s+del\s+alcance",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------


def cargar_json(ruta: Path, defecto):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else defecto


def guardar_json(ruta: Path, datos) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def texto_para_ragas(v: VeredictoMulta, con_recomendaciones: bool = False) -> str:
    """Extrae del JSON estricto la justificacion en prosa que Ragas evalua."""
    if v.clasificacion == Clasificacion.FUERA_DE_ALCANCE:
        return v.respuesta_fuera_de_alcance or v.fundamento
    partes = [v.fundamento, *(h.explicacion for h in v.hallazgos)]
    if con_recomendaciones:
        partes += v.recomendaciones
    return " ".join(p.strip() for p in partes if p and p.strip())


def con_reintento(fn, intentos: int = 6):
    """Reintenta ante 429 respetando el 'retry in Ns' que devuelve la API."""
    for i in range(intentos):
        try:
            return fn()
        except AssistantError as exc:
            if i == intentos - 1 or not re.search(r"429|RESOURCE_EXHAUSTED|quota", str(exc), re.I):
                raise
            m = re.search(r"retry in ([\d.]+)s", str(exc), re.I)
            espera = float(m.group(1)) + 2 if m else 30
            print(f"    (cuota agotada, esperando {espera:.0f}s)")
            time.sleep(espera)


def articulo_num(articulo: str) -> str:
    m = re.match(r"\d+", articulo)
    return m.group() if m else articulo


# --------------------------------------------------------------------------
# Paso 1: generar respuestas con el RAG
# --------------------------------------------------------------------------


def generar(config: AppConfig, dataset: list[dict], tag: str, pausa: float, con_rec: bool) -> list[dict]:
    ruta = RUNS / f"{tag}_respuestas.json"
    cache: dict[str, dict] = cargar_json(ruta, {})
    asistente = MultaClaraAssistant(config)

    for item in dataset:
        clave = str(item["id"])
        if clave in cache:
            continue
        print(f"[{item['id']:>2}/{len(dataset)}] {item['pregunta'][:70]}")
        asistente.reiniciar()  # cada pregunta es independiente
        r = con_reintento(lambda: asistente.consultar(item["pregunta"]))
        v = r.veredicto
        cache[clave] = {
            "id": item["id"],
            "tipo": item["tipo"],
            "pregunta": item["pregunta"],
            "ground_truth": item["ground_truth"],
            "articulos_esperados": item.get("articulos", []),
            "respuesta": texto_para_ragas(v, con_rec),
            "clasificacion": v.clasificacion.value,
            "contextos": [f"[{f.cita}] {f.texto}" for f in r.fuentes],
            "articulos_recuperados": sorted({articulo_num(f.articulo) for f in r.fuentes}),
            "citas_no_respaldadas": r.citas_no_respaldadas,
            "veredicto": v.model_dump(mode="json"),
        }
        guardar_json(ruta, cache)
        time.sleep(pausa)

    return [cache[str(i["id"])] for i in dataset if str(i["id"]) in cache]


# --------------------------------------------------------------------------
# Paso 2: Ragas
# --------------------------------------------------------------------------


def evaluar_con_ragas(config: AppConfig, registros: list[dict], tag: str, juez: str, rpm: float, lote: int) -> dict:
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    answer_relevancy.strictness = 1  # Gemini no admite n>1 candidatos por llamada

    limitador = InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.2, max_bucket_size=1)

    def crear_modelos():
        # transport="rest": el cliente asincrono gRPC queda atado al event loop
        # del primer lote (Ragas abre uno nuevo por evaluate) y falla en los demas
        # con "Event loop is closed". Se recrea ademas en cada lote.
        llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=juez, google_api_key=config.api_key, temperature=0,
                rate_limiter=limitador, transport="rest",
            )
        )
        emb = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(
                model=f"models/{config.embedding_model}", google_api_key=config.api_key, transport="rest",
            )
        )
        return llm, emb

    metricas = [faithfulness, answer_relevancy, context_precision, context_recall]
    run_config = RunConfig(max_workers=2, timeout=240, max_retries=8, max_wait=60)

    ruta = RUNS / f"{tag}_ragas.json"
    cache: dict[str, dict] = cargar_json(ruta, {})
    pendientes = [r for r in registros if str(r["id"]) not in cache or any(cache[str(r["id"])].get(m) is None for m in METRICAS)]
    print(f"Ragas: {len(pendientes)} preguntas por evaluar (juez: {juez}, ~{rpm:g} llamadas/min)")

    for i in range(0, len(pendientes), lote):
        grupo = pendientes[i : i + lote]
        ds = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=r["pregunta"],
                    response=r["respuesta"],
                    retrieved_contexts=r["contextos"] or ["(sin contexto)"],
                    reference=r["ground_truth"],
                )
                for r in grupo
            ]
        )
        llm, emb = crear_modelos()
        resultado = evaluate(ds, metrics=metricas, llm=llm, embeddings=emb, run_config=run_config,
                             show_progress=False, raise_exceptions=False)
        df = resultado.to_pandas()
        for r, (_, fila) in zip(grupo, df.iterrows()):
            cache[str(r["id"])] = {
                m: (None if m not in fila or fila[m] is None or (isinstance(fila[m], float) and math.isnan(fila[m])) else float(fila[m]))
                for m in METRICAS
            }
        guardar_json(ruta, cache)
        vacias = sum(all(cache[str(r["id"])][m] is None for m in METRICAS) for r in grupo)
        aviso = f"  [!] {vacias} sin puntaje (se reintentan al repetir el comando)" if vacias else ""
        print(f"  evaluadas {min(i + lote, len(pendientes))}/{len(pendientes)}{aviso}")
    return cache


# --------------------------------------------------------------------------
# Paso 3: resumen
# --------------------------------------------------------------------------


def promedio(valores: list) -> float | None:
    v = [x for x in valores if x is not None]
    return sum(v) / len(v) if v else None


def rechazo_correcto(r: dict) -> bool:
    """Control de alucinacion en preguntas fuera de alcance / sin respuesta en el corpus."""
    if r["clasificacion"] == Clasificacion.FUERA_DE_ALCANCE.value:
        return True
    if r["tipo"] == "sin_respuesta_corpus":
        v = r["veredicto"]
        texto = " ".join([v["fundamento"], *(h["explicacion"] for h in v["hallazgos"])])
        return bool(_RE_NO_CUBRE.search(texto)) and not r["citas_no_respaldadas"]
    return False


def calcular_resumen(registros: list[dict], ragas: dict) -> dict:
    dentro = [r for r in registros if r["tipo"] == "dentro_corpus"]
    control = [r for r in registros if r["tipo"] != "dentro_corpus"]
    res = {"n_dentro": len(dentro), "n_control": len(control)}
    for m in METRICAS:
        res[m] = promedio([ragas.get(str(r["id"]), {}).get(m) for r in dentro])
        res[f"{m}_todas"] = promedio([ragas.get(str(r["id"]), {}).get(m) for r in registros])
    esperados = [r for r in dentro if r["articulos_esperados"]]
    hits = [any(a in r["articulos_recuperados"] for a in r["articulos_esperados"]) for r in esperados]
    res["hit_articulo"] = sum(hits) / len(hits) if hits else None
    res["rechazo_correcto"] = sum(rechazo_correcto(r) for r in control) / len(control) if control else None
    res["citas_no_respaldadas"] = sum(len(r["citas_no_respaldadas"]) for r in registros)
    res["preguntas_con_cita_no_respaldada"] = sum(bool(r["citas_no_respaldadas"]) for r in registros)
    return res


def fmt(x) -> str:
    return "n/d" if x is None else f"{x:.3f}"


def tabla_markdown(tag: str, res: dict, base_tag: str | None, base: dict | None) -> str:
    filas = [
        ("faithfulness", "Fidelidad al contexto (generacion)"),
        ("answer_relevancy", "Relevancia de la respuesta (generacion)"),
        ("context_precision", "Precision del contexto (recuperacion)"),
        ("context_recall", "Cobertura del contexto (chunking/recuperacion)"),
        ("hit_articulo", "Acierto de articulo en top_k (sin LLM)"),
        ("rechazo_correcto", "Rechazo correcto en preguntas de control"),
    ]
    enc = f"| Metrica | Que mide | `{tag}` |" + (f" `{base_tag}` | Delta |" if base else "")
    sep = "|---|---|---|" + ("---|---|" if base else "")
    out = [f"## Resultados `{tag}` ({res['n_dentro']} preguntas dentro del corpus + {res['n_control']} de control)", "", enc, sep]
    for clave, desc in filas:
        fila = f"| {clave} | {desc} | {fmt(res[clave])} |"
        if base:
            d = None if res[clave] is None or base[clave] is None else res[clave] - base[clave]
            fila += f" {fmt(base[clave])} | {'n/d' if d is None else f'{d:+.3f}'} |"
        out.append(fila)
    out += ["", f"Citas de articulos que no estaban en el contexto recuperado: {res['citas_no_respaldadas']} "
            f"(en {res['preguntas_con_cita_no_respaldada']} respuestas).",
            "Las 4 metricas Ragas se promedian sobre las preguntas *dentro del corpus*; las de control "
            "(fuera de alcance / sin respuesta) se juzgan por el comportamiento de rechazo."]
    return "\n".join(out)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Evalua el RAG de MultaClara con Ragas.")
    ap.add_argument("--tag", required=True, help="nombre de la corrida (p. ej. baseline)")
    ap.add_argument("--collection")
    ap.add_argument("--chroma-dir")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--juez", help="modelo juez de Ragas (por defecto GEMINI_MODEL)")
    ap.add_argument("--rpm-juez", type=float, default=10, help="llamadas/min del juez (plan gratuito)")
    ap.add_argument("--pausa", type=float, default=5, help="segundos entre preguntas al RAG")
    ap.add_argument("--lote", type=int, default=4, help="preguntas por lote de Ragas (checkpoint)")
    ap.add_argument("--limite", type=int, help="usar solo las primeras N preguntas")
    ap.add_argument("--comparar", help="tag de otra corrida para mostrar el delta")
    ap.add_argument("--con-recomendaciones", action="store_true")
    ap.add_argument("--solo-generar", action="store_true")
    ap.add_argument("--solo-resumen", action="store_true")
    args = ap.parse_args()

    try:
        config = AppConfig.from_env()
    except ConfigError as e:
        sys.exit(f"[Error de configuracion] {e}")
    cambios = {"collection": args.collection, "chroma_dir": args.chroma_dir, "top_k": args.top_k}
    config = dataclasses.replace(config, **{k: v for k, v in cambios.items() if v is not None})

    dataset = cargar_json(DATASET, [])
    if args.limite:
        dataset = dataset[: args.limite]
    print(f"Corrida '{args.tag}': coleccion={config.collection} top_k={config.top_k} preguntas={len(dataset)}")

    registros = generar(config, dataset, args.tag, args.pausa, args.con_recomendaciones) if not args.solo_resumen \
        else list(cargar_json(RUNS / f"{args.tag}_respuestas.json", {}).values())
    if args.solo_generar:
        return

    ragas = evaluar_con_ragas(config, registros, args.tag, args.juez or config.model, args.rpm_juez, args.lote) \
        if not args.solo_resumen else cargar_json(RUNS / f"{args.tag}_ragas.json", {})

    res = calcular_resumen(registros, ragas)
    base = None
    if args.comparar:
        reg_b = list(cargar_json(RUNS / f"{args.comparar}_respuestas.json", {}).values())
        base = calcular_resumen(reg_b, cargar_json(RUNS / f"{args.comparar}_ragas.json", {}))
    md = tabla_markdown(args.tag, res, args.comparar, base)
    (RUNS / f"{args.tag}_resumen.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
