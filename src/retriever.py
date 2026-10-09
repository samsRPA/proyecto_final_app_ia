"""Recuperacion: busqueda por similitud sobre el indice ChromaDB."""

from dataclasses import dataclass

import chromadb

from .config import AppConfig
from .embeddings import GeminiEmbedder


class RetrievalError(RuntimeError):
    """El indice no existe, esta vacio o es incompatible con la configuracion."""


@dataclass(frozen=True)
class Fragmento:
    texto: str
    documento: str
    articulo: str
    cita: str
    titulo: str
    capitulo: str
    score: float  # similitud coseno (1 = identico)


def abrir_cliente(config: AppConfig) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=config.chroma_dir)


class Retriever:
    def __init__(self, config: AppConfig, embedder: GeminiEmbedder | None = None):
        self._config = config
        self._embedder = embedder or GeminiEmbedder(config)
        try:
            self._col = abrir_cliente(config).get_collection(config.collection)
        except Exception as exc:
            raise RetrievalError(
                f"No existe el indice '{config.collection}' en '{config.chroma_dir}'. "
                "Ejecuta primero: python -m src.ingest"
            ) from exc
        modelo = (self._col.metadata or {}).get("embedding_model")
        if modelo and modelo != config.embedding_model:
            raise RetrievalError(
                f"El indice se creo con '{modelo}' pero la config usa "
                f"'{config.embedding_model}'. Reindexa o corrige EMBEDDING_MODEL."
            )

    def buscar(self, consulta: str, top_k: int | None = None) -> list[Fragmento]:
        k = top_k or self._config.top_k
        res = self._col.query(
            query_embeddings=[self._embedder.embed_consulta(consulta)],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        fragmentos = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            score = 1.0 - dist  # espacio coseno: distancia = 1 - similitud
            if score < self._config.min_score:
                continue
            fragmentos.append(
                Fragmento(
                    texto=doc,
                    documento=meta.get("documento", ""),
                    articulo=meta.get("articulo", ""),
                    cita=meta.get("cita", ""),
                    titulo=meta.get("titulo", ""),
                    capitulo=meta.get("capitulo", ""),
                    score=score,
                )
            )
        return fragmentos


def formatear_contexto(fragmentos: list[Fragmento]) -> str:
    """Serializa los fragmentos como bloque delimitado para el prompt."""
    if not fragmentos:
        return "(No se recuperaron fragmentos relevantes de la base normativa.)"
    return "\n".join(
        f'<fragmento id="{i}" fuente="{f.cita}">\n{f.texto}\n</fragmento>'
        for i, f in enumerate(fragmentos, start=1)
    )
