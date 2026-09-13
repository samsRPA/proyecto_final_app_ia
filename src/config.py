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

    @classmethod
    def from_env(cls) -> "AppConfig":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ConfigError(
                "Falta GEMINI_API_KEY. Copia .env.example a .env y agrega tu clave "
                "(consíguela en https://aistudio.google.com/apikey)."
            )

        return cls(
            api_key=api_key,
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.2")),
            max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1536")),
        )
