"""Componente de barra lateral (Sidebar) para LabGuard AI."""

import streamlit as st

def render_sidebar(default_confidence: float, default_size: int) -> dict:
    """Renderiza el panel lateral y devuelve la configuración del usuario."""
    with st.sidebar:
        st.title("⚙️ Configuración")
        st.markdown("---")
        
        # Selector de modelo (Preparado para la transición al definitivo)
        modelo_seleccionado = st.selectbox(
            "Seleccionar Modelo YOLOv8",
            options=["Provisional (Pruebas)", "Definitivo (6 Clases)"],
            index=0,
            help="El modelo definitivo estará disponible al finalizar el entrenamiento."
        )
        
        if modelo_seleccionado == "Definitivo (6 Clases)":
            st.warning("⚠️ El modelo definitivo aún no está integrado. Se usará el provisional en modo compatibilidad.")
        else:
            st.success("✅ Modelo provisional activo.")
            
        st.markdown("### Ajustes de Inferencia")
        
        confidence_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.10,
            max_value=0.60,
            value=float(default_confidence),
            step=0.01,
            help="Un valor menor recupera más candidatos, pero puede aumentar falsos positivos."
        )
        
        image_size = st.slider(
            "Tamaño de inferencia",
            min_value=512,
            max_value=640,
            value=default_size,
            step=32,
            help="Una imagen mayor puede ayudar con EPP pequeños, a costa de más tiempo en CPU."
        )
        
        st.markdown("---")
        st.markdown("### Opciones de Análisis")
        
        individual_analysis_enabled = st.toggle(
            "Análisis individual por persona",
            value=False,
            help="Detecta personas con yolov8n.pt y ejecuta el modelo EPP dentro de cada recorte. Puede ser más lento en CPU."
        )
        
        warnings_enabled = st.toggle(
            "Mostrar advertencias visuales",
            value=False,
            help="Solo se dibujan cuando existe una boca localizada de forma fiable y evidencia de ausencia de tapabocas."
        )
        
        return {
            "modelo": modelo_seleccionado,
            "confianza": confidence_threshold,
            "tamano": image_size,
            "analisis_individual": individual_analysis_enabled,
            "advertencias": warnings_enabled
        }
