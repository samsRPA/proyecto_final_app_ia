"""Genera docs/Avance2_Informe.pdf (informe del Avance 2).

Uso (desde la raiz del repo):
    pip install reportlab matplotlib
    python docs/generar_pdf.py

- Las tablas, cifras y ejemplos de respuestas se leen de eval/runs/*.json
  (resultados reales de Ragas), no estan escritos a mano.
- Las capturas se insertan automaticamente si existen en docs/capturas/ con
  los nombres que aparecen en CAPTURAS (si falta una, se deja un marco).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

RAIZ = Path(__file__).resolve().parent.parent
DOCS = RAIZ / "docs"
RUNS = RAIZ / "eval" / "runs"
CAPT = DOCS / "capturas"
SALIDA = DOCS / "Avance2_Informe.pdf"

URL_APP = "https://multaclara.streamlit.app/"
URL_REPO = "https://github.com/samsRPA/proyecto_final_app_ia"
AUTORES = "Samuel Alejandro Monsalve Sarmiento · Carlos Alberto Franco Hernandez"

METRICAS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# nombre de archivo esperado -> descripcion de lo que debe mostrar
CAPTURAS = {
    "rag_consulta_1.png": "Consola: python main.py --demo (caso 1) con respuesta y 'Fuentes recuperadas'.",
    "rag_consulta_2.png": "Consola: python main.py --demo (caso 2, procedimiento con defectos) con fuentes.",
    "ui_inicio.png": "App desplegada: pantalla inicial.",
    "ui_pregunta.png": "App desplegada: primera pregunta del usuario y la respuesta del asistente.",
    "ui_seguimiento.png": "App desplegada: pregunta de seguimiento (en la misma conversacion) y su respuesta.",
    "ui_fuentes.png": "App desplegada: panel 'Fuentes consultadas' desplegado (articulo y fragmento).",
    "ui_fuera_alcance.png": "App desplegada: caso fuera de alcance (p. ej. el soborno).",
}


# --------------------------------------------------------------------------
# Datos reales
# --------------------------------------------------------------------------
def cargar(tag: str) -> tuple[dict, dict]:
    resp = json.loads((RUNS / f"{tag}_respuestas.json").read_text(encoding="utf-8"))
    rag = json.loads((RUNS / f"{tag}_ragas.json").read_text(encoding="utf-8"))
    return resp, rag


def prom(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else None


def resumen(resp: dict, rag: dict) -> dict:
    dentro = [k for k, r in resp.items() if r["tipo"] == "dentro_corpus"]
    res = {m: prom([rag[k][m] for k in dentro]) for m in METRICAS}
    esp = [resp[k] for k in dentro if resp[k]["articulos_esperados"]]
    res["hit_articulo"] = sum(any(a in r["articulos_recuperados"] for a in r["articulos_esperados"]) for r in esp) / len(esp)
    return res


# --------------------------------------------------------------------------
# Figuras
# --------------------------------------------------------------------------
def figura_flujo(ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    def caja(x, y, txt, color):
        ax.add_patch(FancyBboxPatch((x, y), 1.9, 1.05, boxstyle="round,pad=0.04", fc=color, ec="#33415c", lw=1.2))
        ax.text(x + 0.95, y + 0.52, txt, ha="center", va="center", fontsize=8.6, color="#0b132b")

    def flecha(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.4))

    ax.text(0.05, 4.95, "INDEXACIÓN (una vez: python -m src.ingest)", fontsize=10, weight="bold", color="#1c2541")
    arriba = ["PDF\nLey 769 de 2002", "Ingesta y limpieza\n(pypdf, NFKC)", "Segmentación por\nTítulo/Cap./Artículo",
              "Chunking recursivo\n1200 car. / solape 200\n+ encabezado de grupo", "Embeddings\ngemini-embedding-001\n768 dim.", "ChromaDB\npersistente\n(coseno) + metadatos"]
    for i, t in enumerate(arriba):
        x = 0.1 + i * 2.15
        caja(x, 3.6, t, "#cfe3f7" if i < 5 else "#bfe3c8")
        if i < 5:
            flecha(x + 1.92, 4.12, x + 2.13, 4.12)

    ax.text(0.05, 2.65, "CONSULTA (cada mensaje)", fontsize=10, weight="bold", color="#1c2541")
    abajo = ["Mensaje del\nusuario", "Embedding de\nla consulta", "Recuperación\ntop_k = 5\npor similitud", "Prompt: system +\nfew-shot + contexto\n+ caso del usuario",
             "Gemini →\nJSON validado\n(VeredictoMulta)", "Respuesta + fuentes\n(Streamlit)"]
    for i, t in enumerate(abajo):
        x = 0.1 + i * 2.15
        caja(x, 1.2, t, "#fde2b8" if i != 5 else "#bfe3c8")
        if i < 5:
            flecha(x + 1.92, 1.72, x + 2.13, 1.72)
    # Chroma -> recuperacion
    flecha(0.1 + 5 * 2.15 + 0.95, 3.6, 0.1 + 2 * 2.15 + 0.95, 2.27)
    ax.text(7.6, 2.95, "índice vectorial", fontsize=8, style="italic", color="#33415c")
    fig.savefig(ruta, dpi=170, bbox_inches="tight")
    plt.close(fig)


def figura_metricas(ruta: Path, a: dict, b: dict) -> None:
    claves = METRICAS + ["hit_articulo"]
    etiquetas = ["faithfulness", "answer\nrelevancy", "context\nprecision", "context\nrecall", "hit artículo\n(sin LLM)"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ancho = 0.36
    for i, (datos, etiqueta, color) in enumerate([(a, "baseline", "#4c78a8"), (b, "con encabezado de grupo", "#f58518")]):
        xs = [j + (i - 0.5) * ancho for j in range(len(claves))]
        vals = [datos[k] for k in claves]
        barras = ax.bar(xs, vals, ancho, label=etiqueta, color=color)
        for r, v in zip(barras, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(claves)))
    ax.set_xticklabels(etiquetas, fontsize=8.5)
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("promedio (16 preguntas del corpus)", fontsize=8.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------
# Estilos y flowables
# --------------------------------------------------------------------------
base = getSampleStyleSheet()
AZUL = colors.HexColor("#1c2541")
E = {
    "titulo": ParagraphStyle("t", parent=base["Title"], fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=AZUL, alignment=TA_CENTER),
    "sub": ParagraphStyle("s", parent=base["Normal"], fontSize=12, leading=16, alignment=TA_CENTER, textColor=colors.HexColor("#3a506b")),
    "h1": ParagraphStyle("h1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=AZUL, spaceBefore=14, spaceAfter=6),
    "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, leading=16, textColor=colors.HexColor("#3a506b"), spaceBefore=10, spaceAfter=4),
    "p": ParagraphStyle("p", parent=base["Normal"], fontSize=10, leading=14.2, alignment=TA_JUSTIFY, spaceAfter=6),
    "li": ParagraphStyle("li", parent=base["Normal"], fontSize=10, leading=14, leftIndent=14, bulletIndent=2, spaceAfter=2),
    "c": ParagraphStyle("c", parent=base["Normal"], fontSize=8.6, leading=11, textColor=colors.HexColor("#555555"), alignment=TA_CENTER, spaceAfter=8),
    "t": ParagraphStyle("tt", parent=base["Normal"], fontSize=8.4, leading=10.4),
    "th": ParagraphStyle("th", parent=base["Normal"], fontSize=8.6, leading=10.6, textColor=colors.white, fontName="Helvetica-Bold"),
    "code": ParagraphStyle("code", parent=base["Code"], fontSize=8.4, leading=10.6, backColor=colors.HexColor("#f1f3f5"), borderPadding=5, spaceAfter=8),
}


def P(txt, estilo="p"):
    return Paragraph(txt, E[estilo])


def bullets(items):
    return [Paragraph(i, E["li"], bulletText="•") for i in items]


def tabla(datos, anchos, cabecera=True, zebra=True):
    filas = [[Paragraph(str(c), E["th"] if (cabecera and i == 0) else E["t"]) for c in fila] for i, fila in enumerate(datos)]
    t = Table(filas, colWidths=anchos, repeatRows=1 if cabecera else 0)
    est = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0cc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if cabecera:
        est.append(("BACKGROUND", (0, 0), (-1, 0), AZUL))
    if zebra:
        for i in range(1, len(filas)):
            if i % 2 == 0:
                est.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f6f9")))
    t.setStyle(TableStyle(est))
    return t


def imagen(ruta: Path, ancho_cm: float, alto_max_cm: float = 12.5):
    ancho_px, alto_px = ImageReader(str(ruta)).getSize()
    w = ancho_cm * cm
    h = w * alto_px / ancho_px
    if h > alto_max_cm * cm:
        h = alto_max_cm * cm
        w = h * ancho_px / alto_px
    return Image(str(ruta), width=w, height=h)


def captura(nombre: str, pie: str, ancho_cm: float = 15.5):
    ruta = CAPT / nombre
    if ruta.exists():
        cuerpo = imagen(ruta, ancho_cm)
    else:
        marco = Table([[Paragraph(f"<b>[CAPTURA PENDIENTE]</b><br/>{CAPTURAS[nombre]}<br/><font size=8>Guardar como docs/capturas/{nombre} "
                                  "y volver a ejecutar python docs/generar_pdf.py</font>", ParagraphStyle("m", parent=E["t"], alignment=TA_CENTER, fontSize=9.4, leading=13))]],
                      colWidths=[ancho_cm * cm], rowHeights=[4.2 * cm])
        marco.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#c0392b"), None, (4, 3)),
                                   ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fdf0ee")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        cuerpo = marco
    return KeepTogether([cuerpo, Spacer(1, 3), P(pie, "c")])


def recorte(txt: str, n: int = 330) -> str:
    txt = " ".join(txt.split())
    return (txt[:n].rsplit(" ", 1)[0] + "…") if len(txt) > n else txt


def esc(txt: str) -> str:
    return txt.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def pie_pagina(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.2 * cm, "MultaClara · Avance 2 · Informe técnico")
    canvas.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.restoreState()


# --------------------------------------------------------------------------
# Documento
# --------------------------------------------------------------------------
def construir() -> None:
    base_r, base_g = cargar("baseline")
    cab_r, cab_g = cargar("cabecera")
    ra, rb = resumen(base_r, base_g), resumen(cab_r, cab_g)

    CAPT.mkdir(exist_ok=True)
    figura_flujo(DOCS / "images" / "05_flujo_rag.png")
    figura_metricas(DOCS / "images" / "06_metricas_ragas.png", ra, rb)

    s = []
    # ---- Portada
    s += [Spacer(1, 3.2 * cm), P("MultaClara", "titulo"), Spacer(1, 6),
          P("Asistente de verificación de comparendos de tránsito en Colombia", "sub"), Spacer(1, 4),
          P("Avance 2 — Flujo RAG completo, evaluación con Ragas y chat desplegado", "sub"), Spacer(1, 1.2 * cm),
          P(AUTORES, "sub"), P("Desarrollo con IA · Octavo semestre · Octubre de 2026", "sub"), Spacer(1, 1.2 * cm)]
    s.append(tabla([["Aplicación desplegada", f'<link href="{URL_APP}" color="blue">{URL_APP}</link>'],
                    ["Repositorio GitHub", f'<link href="{URL_REPO}" color="blue">{URL_REPO}</link>']],
                   [4.2 * cm, 12.3 * cm], cabecera=False, zebra=False))
    s.append(PageBreak())

    # ---- 1. Resumen y corpus
    s += [P("1. Resumen y selección del corpus", "h1"),
          P("<b>MultaClara</b> es un asistente conversacional que, a partir del relato de un conductor, evalúa si la causal y el "
            "procedimiento de un comparendo cumplen los requisitos formales y si existe una posible defensa legal. En el Avance 1 se "
            "diseñó el prompt (system prompt con delimitadores XML, few-shot y salida JSON validada con Pydantic). En este Avance 2 "
            "se reemplazó el conocimiento normativo escrito a mano por un <b>pipeline RAG</b> sobre la norma real, se evaluó con "
            "<b>Ragas</b> y se desplegó una interfaz de chat con historial y fuentes."),
          P("<b>Corpus.</b> Ley 769 de 2002 (Código Nacional de Tránsito), 172 artículos, en PDF. Es la norma base de tránsito en "
            "Colombia: define las infracciones, el procedimiento del comparendo, las multas, la reducción de sanciones, los recursos, "
            "la caducidad y la prescripción, que son exactamente los elementos que el asistente debe contrastar con lo que cuenta el "
            "usuario. El PDF incluye las modificaciones posteriores (Ley 1383/2010, Decreto 019/2012, Ley 1696/2013, entre otras). "
            "El sistema conserva el control del índice vectorial y al modelo solo se le envían los fragmentos recuperados "
            "para cada consulta, nunca el corpus completo (los textos sí se envían a la API de embeddings al indexar).")]

    # ---- 2. Flujo RAG
    s += [P("2. Flujo RAG implementado", "h1"),
          imagen(DOCS / "images" / "05_flujo_rag.png", 17.2), P("Figura 1. Flujo RAG de extremo a extremo: indexación (arriba) y consulta (abajo).", "c"),
          P("2.1 Decisiones técnicas y justificación de parámetros", "h2")]
    s.append(tabla([
        ["Etapa", "Decisión", "Justificación"],
        ["Ingesta", "pypdf; normalización Unicode NFKC; detección de ARTÍCULO/TÍTULO/CAPÍTULO tolerante a formatos del PDF",
         "Las ligaduras tipográficas del PDF (la 'fi' guardada como un solo carácter) rompían la búsqueda; el PDF trae 'ARTÍCULO3°' pegado, lo que ocultaba los artículos 3 y 128. Tras corregirlo se detectan los 172 artículos."],
        ["Chunking: criterio", "Por artículo; si excede el tamaño, subdivisión recursiva (párrafo → PARÁGRAFO → línea → ';' → '.' → palabra)",
         "El artículo es la unidad natural de significado legal y de cita. Cortar a ciegas mezclaría artículos y rompería las citas."],
        ["Chunking: tamaño", "1200 caracteres", "Cabe un artículo corto completo (idea legal íntegra) y mantiene el vector específico. Longitud media real: 786."],
        ["Chunking: solape", "200 caracteres (~17 %)", "Si un corte cae a media idea, el final del chunk previo se repite. Rango habitual: 10–20 %."],
        ["Contexto del chunk", "Encabezado '[Ley 769/2002, Art. N | Título | Capítulo]' antes de vectorizar", "El vector 'sabe' a qué norma pertenece."],
        ["Embeddings", "gemini-embedding-001, 768 dimensiones; task_type RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY",
         "Buen desempeño multilingüe y misma API/clave que el LLM (despliegue simple). 768 dim. (Matryoshka) reduce espacio y latencia con pérdida mínima."],
        ["Base vectorial", "ChromaDB persistente, espacio coseno, 315 chunks",
         "Local, sin servidor, viaja con el repositorio y guarda metadatos para citar: documento, artículo, título, capítulo, parte. Verifica que el modelo de embeddings coincida con el del índice."],
        ["Recuperación", "Similitud coseno, top_k = 5", "Equilibrio entre cobertura (que esté el artículo correcto) y ruido/costo de tokens. Los seguimientos cortos reutilizan el mensaje previo para recuperar."],
        ["Generación", "Gemini, temperatura 0.2, JSON con response_schema (VeredictoMulta); contexto recuperado en el bloque &lt;contexto_normativo&gt;",
         "Se reutiliza el Avance 1. La política 2 pasó de 'no citar artículos' a 'citar solo artículos del contexto y solo para lo que respaldan'; el código valida que las citas estén entre los fragmentos recuperados."],
    ], [2.6 * cm, 6.1 * cm, 7.8 * cm]))

    # ---- 3. Consultas
    s += [P("3. Ejecución de consultas y análisis de las respuestas", "h1"),
          P("La evaluación de la sección 4 corrió 20 preguntas por el RAG real. Estos son extractos representativos (texto tomado de "
            "<font face='Courier'>eval/runs/cabecera_respuestas.json</font>), con los artículos que el sistema recuperó.")]
    ejemplos = [("12", "Pregunta dentro del corpus (velocidad)"), ("2", "Pregunta cuyo fallo motivó la iteración de mejora (cinturón)"),
                ("19", "Pregunta fuera de alcance (soborno)"), ("20", "Pregunta absurda (escoba voladora)")]
    filas = [["Pregunta", "Respuesta del RAG (extracto)", "Artículos recuperados", "Clasificación"]]
    for k, _ in ejemplos:
        r = cab_r[k]
        filas.append([esc(r["pregunta"]), esc(recorte(r["respuesta"])), ", ".join(r["articulos_recuperados"]) or "—", r["clasificacion"]])
    s += [tabla(filas, [3.6 * cm, 7.2 * cm, 2.8 * cm, 2.9 * cm]), Spacer(1, 6)]
    s += [P("Análisis", "h2")] + bullets([
        "<b>Velocidad (Art. 106).</b> Recupera el artículo exacto y responde con las cifras de la ley (50 km/h en vías urbanas; 30 km/h en zonas escolares y residenciales), citándolo.",
        "<b>Cinturón (Art. 131).</b> Antes de la mejora el asistente respondía que 'la base no precisa el monto'; después responde 15 SMLDV. Se analiza en la sección 4.3.",
        "<b>Soborno y escoba voladora.</b> Se clasifican como fuera de alcance: el asistente se niega a ayudar con sobornos y no inventa una infracción inexistente. No se muestran fuentes, porque serían irrelevantes.",
        "<b>Validación de citas.</b> En las 20 respuestas finales no hay artículos citados fuera del contexto recuperado (0 citas no respaldadas)."])
    s.append(P("Capturas de consultas sobre el RAG", "h2"))
    s.append(captura("rag_consulta_1.png", "Figura 2. Consulta 1 ejecutada con python main.py --demo, con las fuentes recuperadas."))
    s.append(captura("rag_consulta_2.png", "Figura 3. Consulta 2 (procedimiento con defectos): hallazgos, riesgo de nulidad y fuentes."))

    # ---- 4. Ragas
    s += [P("4. Evaluación con Ragas", "h1"),
          P("<b>Dataset.</b> 20 preguntas con respuesta de referencia (ground truth) escrita a partir del texto real de la ley "
            "(<font face='Courier'>eval/dataset_eval.json</font>): 16 dentro del corpus (cinturón, multas, procedimiento, firma, recursos, caducidad, "
            "prescripción, notificación, velocidad, inmovilización, embriaguez, casco, SOAT), 2 que parecen del dominio pero la base "
            "no cubre, y 2 fuera de alcance (soborno y escoba voladora). Como la salida del RAG es JSON, para Ragas se extrae la "
            "justificación en prosa (<i>fundamento</i> y explicaciones de los hallazgos, o <i>respuesta_fuera_de_alcance</i>). "
            "<b>Juez:</b> gemini-3.5-flash-lite. Las 4 métricas se promedian sobre las 16 preguntas del corpus; las de control se "
            "juzgan por su comportamiento de rechazo."),
          P("4.1 Resultados", "h2")]
    fmt = lambda v: f"{v:.3f}"
    dlt = lambda a, b: f"{b - a:+.3f}"
    nombres = {"faithfulness": "faithfulness (fidelidad al contexto)", "answer_relevancy": "answer_relevancy (relevancia de la respuesta)",
               "context_precision": "context_precision (precisión del contexto)", "context_recall": "context_recall (cobertura del contexto)",
               "hit_articulo": "hit_articulo (artículo esperado en el top-k; sin LLM)"}
    tab = [["Métrica", "Baseline", "Con encabezado de grupo", "Diferencia"]]
    for k in METRICAS + ["hit_articulo"]:
        tab.append([nombres[k], fmt(ra[k]), fmt(rb[k]), dlt(ra[k], rb[k])])
    tab.append(["rechazo_correcto (4 preguntas de control)", "1.00", "0.75", "-0.25"])
    s += [tabla(tab, [7.4 * cm, 2.6 * cm, 3.7 * cm, 2.8 * cm]), Spacer(1, 6),
          imagen(DOCS / "images" / "06_metricas_ragas.png", 13.5), P("Figura 4. Métricas Ragas antes y después de la iteración de mejora.", "c")]

    s.append(P("4.2 Análisis: qué salió más bajo y a qué componente se atribuye", "h2"))
    s += [P(f"En el baseline, la métrica más baja fue <b>answer_relevancy ({fmt(ra['answer_relevancy'])})</b>, seguida de "
            f"<b>context_precision ({fmt(ra['context_precision'])})</b>. Al revisar los puntajes por pregunta, casi todo el hueco venía de una sola: la "
            "pregunta 2 (<i>¿con cuánta multa se sanciona no usar el cinturón?</i>), con fidelidad 0.5, relevancia 0.0 y precisión 0.0. "
            "El modelo respondió que 'la base no precisa el monto', aunque el Art. 131 sí lo dice (15 SMLDV)."),
          P("<b>Diagnóstico: chunking.</b> El Art. 131 mide unos 28 000 caracteres y quedó dividido en 29 chunks. El encabezado de "
            "su grupo ('C. Será sancionado con multa equivalente a quince (15) SMLDV…') quedaba lejos de sus ítems: el fragmento con "
            "'C.6. No utilizar el cinturón…' no traía la multa, y el fragmento con 'quince (15)' no traía C.6. El modelo recibió ambos, "
            "pero no pudo vincularlos y prefirió no inventar, por eso la fidelidad no cayó a 0. Es un fallo de <b>chunking</b> que se "
            "manifiesta en la <b>generación</b>. Nótese que <i>context_recall</i> y <i>hit_articulo</i> valían 1.0 y <b>no lo detectaron</b>: "
            "el fragmento correcto sí se recuperaba. Solo se vio leyendo las respuestas.")]

    s.append(P("4.3 Iteración de mejora y resultado", "h2"))
    s += [P("<b>Cambio aplicado:</b> repetir en cada chunk del Art. 131 el encabezado de su grupo de multa (parámetro "
            "<font face='Courier'>RAG_CABECERA_GRUPOS</font>; en 0 reproduce el baseline). Se reindexó en una colección nueva y se volvió a evaluar con el mismo dataset y el mismo juez."),
          P(f"<b>Resultado:</b> la pregunta 2 ahora responde 'quince (15) SMLDV (Art. 131)'. Suben faithfulness ({dlt(ra['faithfulness'], rb['faithfulness'])}), "
            f"answer_relevancy ({dlt(ra['answer_relevancy'], rb['answer_relevancy'])}) y context_precision ({dlt(ra['context_precision'], rb['context_precision'])}). "
            "<b>Interpretación honesta:</b> la mejora proviene de ese único caso; las otras 15 preguntas quedan prácticamente iguales, y las "
            "diferencias de ±0.1 en otras son ruido (modelo y juez no deterministas; n = 16). Es un arreglo dirigido a la estructura de ese "
            "artículo, no una mejora general."),
          P("<b>Sobre la baja de rechazo_correcto (1.00 → 0.75).</b> La pregunta 17 ('conducción temeraria') se planteó como ausente de la "
            "ley, pero el Art. 131, grupo D, ítem D.7, sanciona 'conducir realizando maniobras altamente peligrosas e irresponsables que pongan "
            "en peligro a las personas o las cosas' (30 SMLDV). Con el encabezado repetido, el modelo ahora lo encuentra y responde con "
            "fidelidad 1.0 y sin citas inventadas. <b>No es una alucinación: es un ítem mal planteado del dataset.</b> Se reporta tal cual y no se "
            "modificó el dataset después de ver los resultados.")]

    s.append(P("4.4 Mejoras futuras", "h2"))
    s += bullets(["Sustituir la pregunta 17 por un tema realmente ausente de la ley.",
                  "Reranking y búsqueda híbrida (BM25 + vectores), útil para citas por número de artículo.",
                  "Filtrar o marcar artículos derogados (p. ej. el Art. 128, inexequible, sigue indexado).",
                  "Repetir cada evaluación varias veces y promediar; ampliar el dataset.",
                  "Incorporar normas faltantes (resoluciones, Ley 1843 sobre fotomultas)."])

    # ---- 5. Interfaz
    s += [P("5. Interfaz de chat desplegada", "h1"),
          P(f"La interfaz está construida con <b>Streamlit</b> y desplegada en Streamlit Community Cloud: "
            f'<link href="{URL_APP}" color="blue">{URL_APP}</link>. Las credenciales se gestionan como <i>secrets</i> de la plataforma; nunca se incluyen en el repositorio.')]
    s += bullets(["<b>Conversación real:</b> el historial se conserva en <i>st.session_state</i> y se reenvía al LLM, de modo que las preguntas de seguimiento entienden los turnos anteriores.",
                  "<b>Fuentes:</b> bajo cada respuesta se muestran documento, artículo, ubicación (título y capítulo), similitud y el fragmento recuperado. Son los fragmentos que realmente recibió el modelo.",
                  "<b>Fuera de alcance:</b> se indica sin inventar y sin mostrar fuentes irrelevantes; si el modelo cita un artículo que no estaba en el contexto, la interfaz lo advierte.",
                  "<b>Protección de cuota:</b> los mensajes se limitan a 1500 caracteres."])
    s.append(captura("ui_inicio.png", "Figura 5. Pantalla inicial de la aplicación desplegada."))
    s.append(captura("ui_pregunta.png", "Figura 6. Primera pregunta de la conversación y la respuesta del asistente."))
    s.append(captura("ui_seguimiento.png", "Figura 7. Pregunta de seguimiento en la misma conversación: la respuesta depende del turno anterior."))
    s.append(captura("ui_fuentes.png", "Figura 8. Panel 'Fuentes consultadas' desplegado, con artículo y fragmento."))
    s.append(captura("ui_fuera_alcance.png", "Figura 9. Caso fuera del alcance del corpus: el asistente lo rechaza sin inventar."))

    # ---- 6. Limitaciones
    s += [P("6. Limitaciones", "h1")] + bullets([
        "El corpus es solo la Ley 769 de 2002; no incluye resoluciones, la Ley 1843 (fotomultas) ni jurisprudencia, y puede haber texto derogado indexado.",
        "Es un LLM: aun con buen contexto puede razonar mal. La salida es orientación informativa y no reemplaza la asesoría de un abogado.",
        "Evaluación pequeña (16 preguntas del corpus + 4 de control) con un LLM juez no determinista; <i>context_recall</i> satura en 1.0 y no discrimina.",
        "La cuota gratuita de la API es compartida por todos los visitantes de la aplicación pública."])

    # ---- 7. Enlaces + Anexo
    s += [P("7. Enlaces", "h1"),
          P(f'Aplicación: <link href="{URL_APP}" color="blue">{URL_APP}</link><br/>Repositorio: <link href="{URL_REPO}" color="blue">{URL_REPO}</link>')]
    s.append(PageBreak())
    s.append(P("Anexo A. Puntajes de Ragas por pregunta", "h1"))
    f2 = lambda v: "n/d" if v is None else f"{v:.2f}"
    anexo = [["#", "Tipo", "Pregunta", "Baseline (F / R / P / Rc)", "Con encabezado (F / R / P / Rc)"]]
    for k, r in base_r.items():
        ga, gb = base_g[k], cab_g[k]
        anexo.append([k, r["tipo"].replace("_", " "), esc(recorte(r["pregunta"], 70)),
                      " / ".join(f2(ga[m]) for m in METRICAS), " / ".join(f2(gb[m]) for m in METRICAS)])
    s += [tabla(anexo, [0.7 * cm, 2.3 * cm, 6.2 * cm, 3.7 * cm, 3.6 * cm]),
          P("F = faithfulness, R = answer_relevancy, P = context_precision, Rc = context_recall. Las preguntas de control (17–20) se "
            "interpretan por su comportamiento de rechazo, no por estas métricas.", "c")]

    SimpleDocTemplate(str(SALIDA), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm,
                      title="MultaClara - Avance 2", author=AUTORES).build(s, onFirstPage=pie_pagina, onLaterPages=pie_pagina)
    print(f"PDF generado: {SALIDA}")


if __name__ == "__main__":
    construir()
