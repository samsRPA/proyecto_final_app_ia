"""Logica de orquestacion: configuracion + RAG + prompts + esquema de salida."""

import re
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from .config import AppConfig
from .prompts import SYSTEM_PROMPT, construir_historial_few_shot, construir_turno_usuario
from .retriever import Fragmento, Retriever, formatear_contexto
from .schemas import VeredictoMulta

DISCLAIMER = (
    "MultaClara ofrece una orientacion informativa inicial y no reemplaza "
    "una asesoria juridica formal. Verifica cualquier plazo o recurso con "
    "el organismo de transito competente o un abogado."
)

# Mensajes de seguimiento cortos ("¿y si no pago?") no tienen contenido
# suficiente para recuperar; se les antepone el turno previo del usuario.
_UMBRAL_PALABRAS_SEGUIMIENTO = 15


_RE_CITA = re.compile(r"\b(?:art[íi]culos?|art)\.?\s*(\d+)", re.IGNORECASE)


_RE_NOTA = re.compile(r"modificad|adicionad|sustituid|derogad|reglamentad|declarad", re.IGNORECASE)


def articulos_citados(veredicto: VeredictoMulta) -> set[str]:
    """Numeros de articulo mencionados en el texto libre del veredicto."""
    textos = [veredicto.fundamento, veredicto.resumen_caso, *veredicto.recomendaciones]
    textos += [h.explicacion for h in veredicto.hallazgos]
    citados = set()
    for t in textos:
        for m in _RE_CITA.finditer(t):
            # "Modificado por el art. 22, Ley 1383" es una nota de la norma, no una cita al Codigo.
            if _RE_NOTA.search(t[max(0, m.start() - 40) : m.start()]):
                continue
            citados.add(m.group(1))
    return citados


class AssistantError(RuntimeError):
    """Error al invocar el modelo o al interpretar su respuesta."""


@dataclass(frozen=True)
class RespuestaRAG:
    veredicto: VeredictoMulta
    fuentes: list[Fragmento] = field(default_factory=list)
    consulta_recuperacion: str = ""
    # Articulos citados por el modelo que NO estaban en los fragmentos recuperados.
    citas_no_respaldadas: list[str] = field(default_factory=list)


class MultaClaraAssistant:
    """Asistente de verificacion de comparendos de transito (RAG)."""

    def __init__(self, config: AppConfig, retriever: Retriever | None = None):
        self._config = config
        self._client = genai.Client(api_key=config.api_key)
        self._retriever = retriever or Retriever(config)
        self._generation_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            response_mime_type="application/json",
            response_schema=VeredictoMulta,
        )
        self._few_shot = construir_historial_few_shot()
        # Turnos reales previos: relato sin contexto + JSON del modelo.
        # El contexto recuperado solo se envia en el turno actual para que
        # el prompt no crezca con fragmentos obsoletos.
        self._turnos: list[tuple[str, str]] = []

    def reiniciar(self) -> None:
        self._turnos.clear()

    def _consulta_de_recuperacion(self, relato: str) -> str:
        if self._turnos and len(relato.split()) < _UMBRAL_PALABRAS_SEGUIMIENTO:
            return f"{self._turnos[-1][0]}\n{relato}"
        return relato

    def _historial(self) -> list[types.Content]:
        historial = list(self._few_shot)
        for relato, json_modelo in self._turnos[-self._config.max_turnos_historial :]:
            historial.append(
                types.Content(role="user", parts=[types.Part(text=construir_turno_usuario(relato))])
            )
            historial.append(types.Content(role="model", parts=[types.Part(text=json_modelo)]))
        return historial

    def consultar(self, relato_usuario: str) -> RespuestaRAG:
        """Recupera contexto, genera el veredicto y devuelve ambos."""
        if not relato_usuario or not relato_usuario.strip():
            raise AssistantError("El relato del usuario no puede estar vacio.")
        relato = relato_usuario.strip()

        consulta = self._consulta_de_recuperacion(relato)
        try:
            fuentes = self._retriever.buscar(consulta)
        except Exception as exc:
            raise AssistantError(f"No fue posible consultar la base normativa: {exc}") from exc

        turno = types.Content(
            role="user",
            parts=[types.Part(text=construir_turno_usuario(relato, formatear_contexto(fuentes)))],
        )
        try:
            respuesta = self._client.models.generate_content(
                model=self._config.model,
                contents=[*self._historial(), turno],
                config=self._generation_config,
            )
        except Exception as exc:  # errores de red/API del SDK
            raise AssistantError(f"No fue posible contactar al modelo: {exc}") from exc

        veredicto = respuesta.parsed
        if veredicto is None:
            raise AssistantError("El modelo no devolvio un JSON valido segun el esquema esperado.")

        self._turnos.append((relato, respuesta.text))
        recuperados = {re.match(r"\d+", f.articulo).group() for f in fuentes if re.match(r"\d+", f.articulo)}
        no_respaldadas = sorted(articulos_citados(veredicto) - recuperados, key=int)
        return RespuestaRAG(
            veredicto=veredicto,
            fuentes=fuentes,
            consulta_recuperacion=consulta,
            citas_no_respaldadas=no_respaldadas,
        )

    def analizar_caso(self, relato_usuario: str) -> VeredictoMulta:
        """Compatibilidad con main.py (Avance 1): solo el veredicto."""
        return self.consultar(relato_usuario).veredicto
