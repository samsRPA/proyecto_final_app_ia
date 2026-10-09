"""Sistema de configuracion del asistente.

Centraliza todo lo que puede variar entre entornos (clave de API, modelo,
parametros de generacion) en un solo lugar, cargado desde variables de
entorno (.env). Nada de esto queda quemado dentro de los prompts.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Error de configuracion (p. ej. falta la API key)."""


@dataclass(frozen=True)
class AppConfig:
    api_key: str
    model: str
    temperature: float
    max_output_tokens: int
    # --- RAG (Avance 2) ---
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768
    data_dir: str = "data/raw"
    chroma_dir: str = "chroma_db"
    collection: str = "exp_cabecera"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    # Repite el encabezado de grupo de multa (Art. 131) en cada chunk. Falso = baseline.
    repetir_cabecera_grupos: bool = True
    # Similitud coseno minima (0-1) para aceptar un fragmento; 0 = sin filtro.
    min_score: float = 0.0
    # Turnos previos (usuario+modelo) que se reenvian al LLM en conversaciones.
    max_turnos_historial: int = 6

    @classmethod
    def from_env(cls, require_api_key: bool = True) -> "AppConfig":
        api_key = os.getenv("GEMINI_API_KEY", "")
        if require_api_key and not api_key:
            raise ConfigError(
                "Falta GEMINI_API_KEY. Copia .env.example a .env y agrega tu clave "
                "(consíguela en https://aistudio.google.com/apikey)."
            )

        return cls(
            api_key=api_key,
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1536")),
            embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            data_dir=os.getenv("RAG_DATA_DIR", "data/raw"),
            chroma_dir=os.getenv("CHROMA_DIR", "chroma_db"),
            collection=os.getenv("CHROMA_COLLECTION", "exp_cabecera"),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            repetir_cabecera_grupos=os.getenv("RAG_CABECERA_GRUPOS", "1") != "0",
            min_score=float(os.getenv("RAG_MIN_SCORE", "0.0")),
            max_turnos_historial=int(os.getenv("RAG_MAX_TURNOS", "6")),
        )
