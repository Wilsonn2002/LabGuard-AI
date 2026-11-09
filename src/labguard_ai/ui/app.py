"""Composicion de la aplicacion Streamlit."""


def run() -> None:
    """Muestra la interfaz inicial sin cargar modelos automaticamente."""
    import streamlit as st

    st.set_page_config(page_title="LabGuard AI", page_icon="LG", layout="wide")
    st.title("LabGuard AI")
    st.write("Verificacion academica de EPP en laboratorios universitarios.")
    st.info("La integracion del modelo y las fuentes de imagen se agregaran posteriormente.")
