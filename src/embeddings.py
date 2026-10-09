"""Cliente de embeddings de Gemini (documentos y consultas).

Se usan `task_type` distintos para indexar (RETRIEVAL_DOCUMENT) y para
buscar (RETRIEVAL_QUERY): el modelo optimiza el vector segun su rol, lo que
mejora la recuperacion frente a usar el mismo tipo en ambos lados.
"""

import re
import time

from google import genai
from google.genai import types

from .config import AppConfig

_BATCH = 25  # lotes chicos: en el plan gratuito cada texto cuenta como 1 solicitud
_REINTENTOS = 8
# Plan gratuito: 100 textos/min. Se dosifica a ~80/min para no rozar el limite.
_TEXTOS_POR_MIN = 80


class EmbeddingError(RuntimeError):
    """Fallo al generar embeddings."""


class GeminiEmbedder:
    def __init__(self, config: AppConfig, client: genai.Client | None = None):
        self._model = config.embedding_model
        self._dim = config.embedding_dim
        self._client = client or genai.Client(api_key=config.api_key)

    def _embed(self, textos: list[str], task_type: str) -> list[list[float]]:
        cfg = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self._dim)
        for intento in range(_REINTENTOS):
            try:
                res = self._client.models.embed_content(
                    model=self._model, contents=textos, config=cfg
                )
                return [e.values for e in res.embeddings]
            except Exception as exc:  # 429 / red
                if intento == _REINTENTOS - 1:
                    raise EmbeddingError(f"Fallo al generar embeddings: {exc}") from exc
                # Si la API indica cuanto esperar ("retry in 27.8s"), se respeta.
                m = re.search(r"retry in ([\d.]+)s", str(exc), re.I)
                espera = float(m.group(1)) + 2 if m else min(2**intento, 30)
                print(f"  (limite de cuota, reintentando en {espera:.0f}s)")
                time.sleep(espera)
        raise EmbeddingError("Fallo al generar embeddings")  # inalcanzable

    def embed_documentos(self, textos: list[str]) -> list[list[float]]:
        vectores: list[list[float]] = []
        for i in range(0, len(textos), _BATCH):
            lote = textos[i : i + _BATCH]
            inicio = time.monotonic()
            vectores.extend(self._embed(lote, "RETRIEVAL_DOCUMENT"))
            print(f"  embeddings {len(vectores)}/{len(textos)}")
            if len(vectores) < len(textos):  # dosifica para respetar la cuota por minuto
                pausa = 60 * len(lote) / _TEXTOS_POR_MIN - (time.monotonic() - inicio)
                if pausa > 0:
                    time.sleep(pausa)
        return vectores

    def embed_consulta(self, texto: str) -> list[float]:
        return self._embed([texto], "RETRIEVAL_QUERY")[0]
