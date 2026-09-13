"""Formato de salida estructurado del asistente MultaClara.

Se define con Pydantic y se pasa como `response_schema` a Gemini
(`response_mime_type="application/json"`), de modo que el modelo esta
obligado a devolver JSON valido con exactamente estos campos, en vez de
depender unicamente de que el prompt lo pida "por las buenas".
"""

from enum import Enum

from pydantic import BaseModel, Field


class Clasificacion(str, Enum):
    INFRACCION_DE_TRANSITO = "infraccion_de_transito"
    CONSULTA_GENERAL = "consulta_general"
    FUERA_DE_ALCANCE = "fuera_de_alcance"


class Cumplimiento(str, Enum):
    CUMPLE = "cumple"
    NO_CUMPLE = "no_cumple"
    NO_DETERMINABLE = "no_determinable"


class NivelRiesgoNulidad(str, Enum):
    ALTO = "alto"
    MEDIO = "medio"
    BAJO = "bajo"
    NO_APLICA = "no_aplica"


class Hallazgo(BaseModel):
    criterio: str = Field(description="Requisito procedimental evaluado, p. ej. 'Identificacion del agente'.")
    cumplido: Cumplimiento
    explicacion: str = Field(description="Justificacion breve basada solo en los hechos narrados por el usuario.")


class VeredictoMulta(BaseModel):
    """Salida obligatoria del asistente para cualquier caso de transito."""

    clasificacion: Clasificacion
    resumen_caso: str = Field(description="Resumen neutral de 1-2 frases de lo que el usuario contó.")
    causal_reportada: str = Field(description="Causal de la infraccion tal como la entendio el usuario.")
    hallazgos: list[Hallazgo] = Field(default_factory=list)
    nivel_riesgo_nulidad: NivelRiesgoNulidad
    tiene_posible_defensa: bool
    fundamento: str = Field(description="Explicacion general y prudente; nunca cita articulos inventados.")
    recomendaciones: list[str] = Field(default_factory=list)
    respuesta_fuera_de_alcance: str | None = Field(
        default=None,
        description="Solo se llena cuando clasificacion es fuera_de_alcance; explica por que no se responde.",
    )
