"""Interfaz de chat de MultaClara (Streamlit).

Ejecutar:  streamlit run app.py

- Historial real: cada sesion conserva sus turnos en `st.session_state` y el
  asistente los reenvia al LLM, de modo que las preguntas de seguimiento
  ("¿y si no tengo la copia?") entienden el contexto previo.
- Fuentes: bajo cada respuesta se muestran los fragmentos recuperados de la
  ley (documento, articulo y texto), no lo que el modelo "dice" haber citado.
- Fuera de alcance: no se muestran fuentes (serian irrelevantes y confusas).
- Secretos: GEMINI_API_KEY se lee de las variables de entorno / `st.secrets`.
"""

import os

import streamlit as st

from src.assistant import DISCLAIMER, AssistantError, MultaClaraAssistant, RespuestaRAG
from src.config import AppConfig, ConfigError
from src.retriever import RetrievalError, Retriever
from src.schemas import Clasificacion, Cumplimiento, NivelRiesgoNulidad

MAX_CARACTERES = 1500  # protege la cuota compartida de la API en un despliegue publico

EJEMPLOS = [
    "Un agente sin uniforme ni carné me puso un comparendo y no me dio copia. ¿Tengo defensa?",
    "¿Con cuánta multa se sanciona a quien no usa el cinturón de seguridad?",
    "¿Qué recursos puedo presentar contra un comparendo?",
    "¿Cuál es la mejor forma de sobornar a un agente de tránsito?",
]

ICONO_HALLAZGO = {
    Cumplimiento.CUMPLE: "✅",
    Cumplimiento.NO_CUMPLE: "❌",
    Cumplimiento.NO_DETERMINABLE: "❔",
}
COLOR_RIESGO = {
    NivelRiesgoNulidad.ALTO: "red",
    NivelRiesgoNulidad.MEDIO: "orange",
    NivelRiesgoNulidad.BAJO: "green",
    NivelRiesgoNulidad.NO_APLICA: "gray",
}

st.set_page_config(page_title="MultaClara", page_icon="🚦", layout="centered")


def _cargar_secrets() -> None:
    """Expone st.secrets (Streamlit Cloud) como variables de entorno."""
    try:
        for clave, valor in st.secrets.items():
            if isinstance(valor, (str, int, float, bool)):
                os.environ.setdefault(clave, str(valor))
    except Exception:  # no hay secrets.toml (ejecucion local con .env)
        pass


@st.cache_resource(show_spinner="Cargando la base normativa…")
def cargar_recursos() -> tuple[AppConfig, Retriever]:
    config = AppConfig.from_env()
    return config, Retriever(config)


def nuevo_asistente() -> MultaClaraAssistant:
    config, retriever = cargar_recursos()
    return MultaClaraAssistant(config, retriever=retriever)


def mostrar_fuentes(r: RespuestaRAG) -> None:
    if not r.fuentes:
        return
    with st.expander(f"📚 Fuentes consultadas ({len(r.fuentes)})"):
        st.caption("Fragmentos de la norma recuperados para esta respuesta, ordenados por similitud.")
        for i, f in enumerate(r.fuentes, start=1):
            st.markdown(f"**{i}. {f.cita}**  ·  similitud {f.score:.2f}")
            ubicacion = " › ".join(x for x in (f.titulo, f.capitulo) if x)
            if ubicacion:
                st.caption(ubicacion)
            st.markdown(f"> {f.texto[:700].replace(chr(10), ' ')}{'…' if len(f.texto) > 700 else ''}")


def mostrar_respuesta(r: RespuestaRAG) -> None:
    v = r.veredicto

    if v.clasificacion == Clasificacion.FUERA_DE_ALCANCE:
        st.info(v.respuesta_fuera_de_alcance or v.fundamento, icon="🚫")
        st.caption(DISCLAIMER)
        return

    st.markdown(v.resumen_caso)
    if v.clasificacion == Clasificacion.INFRACCION_DE_TRANSITO:
        st.markdown(f"**Causal reportada:** {v.causal_reportada}")

    if v.hallazgos:
        st.markdown("**Revisión del procedimiento**")
        for h in v.hallazgos:
            st.markdown(f"{ICONO_HALLAZGO[h.cumplido]} **{h.criterio}** — {h.explicacion}")
        col1, col2 = st.columns(2)
        col1.markdown(
            f"**Riesgo de nulidad:** :{COLOR_RIESGO[v.nivel_riesgo_nulidad]}[{v.nivel_riesgo_nulidad.value}]"
        )
        col2.markdown(f"**¿Posible defensa?** {'Sí' if v.tiene_posible_defensa else 'No'}")

    st.markdown(f"**Fundamento.** {v.fundamento}")

    if v.recomendaciones:
        st.markdown("**Recomendaciones**")
        for rec in v.recomendaciones:
            st.markdown(f"- {rec}")

    if r.citas_no_respaldadas:
        st.warning(
            "Esta respuesta menciona artículos que no estaban entre los fragmentos consultados "
            f"(Art. {', '.join(r.citas_no_respaldadas)}). Verifícalos en la norma antes de usarlos.",
            icon="⚠️",
        )
    mostrar_fuentes(r)
    st.caption(DISCLAIMER)


def main() -> None:
    _cargar_secrets()

    st.title("🚦 MultaClara")
    st.caption(
        "Verifica si un comparendo de tránsito en Colombia cumplió el procedimiento, "
        "con base en la Ley 769 de 2002 (Código Nacional de Tránsito)."
    )

    try:
        cargar_recursos()
    except ConfigError as e:
        st.error(f"Falta configuración: {e}")
        st.stop()
    except RetrievalError as e:
        st.error(f"No se pudo abrir la base normativa: {e}")
        st.stop()

    if "asistente" not in st.session_state:
        st.session_state.asistente = nuevo_asistente()
        st.session_state.mensajes = []  # [{"rol": "user"|"assistant", "texto"|"respuesta": ...}]

    with st.sidebar:
        st.header("Sobre MultaClara")
        st.write(
            "Cuéntame qué pasó cuando te detuvo o multó un agente de tránsito. "
            "Busco en la norma los fragmentos relevantes y te muestro de dónde sale cada respuesta."
        )
        st.write("Puedes hacer preguntas de seguimiento: recuerdo lo que hemos hablado.")
        if st.button("🗑️ Nueva conversación", use_container_width=True):
            st.session_state.asistente.reiniciar()
            st.session_state.mensajes = []
            st.rerun()
        st.divider()
        st.markdown("**Prueba con:**")
        for i, ejemplo in enumerate(EJEMPLOS):
            if st.button(ejemplo, key=f"ej{i}", use_container_width=True):
                st.session_state.pendiente = ejemplo
        st.divider()
        st.caption("Orientación informativa; no reemplaza la asesoría de un abogado.")

    if not st.session_state.mensajes:
        st.chat_message("assistant").markdown(
            "Hola 👋 Soy MultaClara. Cuéntame qué ocurrió con tu comparendo "
            "o pregúntame por una norma de tránsito."
        )

    for m in st.session_state.mensajes:
        with st.chat_message(m["rol"]):
            if m["rol"] == "user":
                st.markdown(m["texto"])
            else:
                mostrar_respuesta(m["respuesta"])

    prompt = st.chat_input("Escribe tu caso o tu pregunta…") or st.session_state.pop("pendiente", None)
    if not prompt:
        return

    if len(prompt) > MAX_CARACTERES:
        st.warning(f"Tu mensaje es muy largo ({len(prompt)} caracteres). Resúmelo a menos de {MAX_CARACTERES}.")
        return

    st.session_state.mensajes.append({"rol": "user", "texto": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Consultando la norma…"):
                respuesta = st.session_state.asistente.consultar(prompt)
        except AssistantError as e:
            st.error(f"No pude responder en este momento: {e}")
            return
        mostrar_respuesta(respuesta)
    st.session_state.mensajes.append({"rol": "assistant", "respuesta": respuesta})


main()
