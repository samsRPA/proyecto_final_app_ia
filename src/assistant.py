"""Logica de orquestacion: une configuracion + prompts + esquema de salida."""

from google import genai
from google.genai import types

from .config import AppConfig
from .prompts import SYSTEM_PROMPT, construir_historial_few_shot, construir_turno_usuario
from .schemas import VeredictoMulta

DISCLAIMER = (
    "MultaClara ofrece una orientacion informativa inicial y no reemplaza "
    "una asesoria juridica formal. Verifica cualquier plazo o recurso con "
    "el organismo de transito competente o un abogado."
)


class AssistantError(RuntimeError):
    """Error al invocar el modelo o al interpretar su respuesta."""


class MultaClaraAssistant:
    """Asistente de verificacion de comparendos de transito."""

    def __init__(self, config: AppConfig):
        self._config = config
        self._client = genai.Client(api_key=config.api_key)
        self._generation_config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            response_mime_type="application/json",
            response_schema=VeredictoMulta,
        )
        self._chat = self._client.chats.create(
            model=config.model,
            config=self._generation_config,
            history=construir_historial_few_shot(),
        )

    def analizar_caso(self, relato_usuario: str) -> VeredictoMulta:
        if not relato_usuario or not relato_usuario.strip():
            raise AssistantError("El relato del usuario no puede estar vacio.")

        mensaje = construir_turno_usuario(relato_usuario)

        try:
            respuesta = self._chat.send_message(mensaje)
        except Exception as exc:  # errores de red/API del SDK
            raise AssistantError(f"No fue posible contactar al modelo: {exc}") from exc

        veredicto = respuesta.parsed
        if veredicto is None:
            raise AssistantError(
                "El modelo no devolvio un JSON valido segun el esquema esperado."
            )
        return veredicto
