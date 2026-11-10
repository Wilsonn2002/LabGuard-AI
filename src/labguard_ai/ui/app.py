"""Interfaz Streamlit para analizar fotografias de laboratorio."""

from pathlib import Path

from labguard_ai.config import ModelConfiguration, load_model_configuration
from labguard_ai.core import EvaluationStatus, PersonAnalysisResult, PpeType
from labguard_ai.detection import YoloPersonDetector, YoloPpeDetector
from labguard_ai.processing.image import ImageAnalyzer
from labguard_ai.processing.person import PersonPpeAnalyzer
from labguard_ai.ui.sidebar import render_sidebar
from labguard_ai.ui.images import (
    annotate_image,
    annotate_people,
    count_detections,
    decode_uploaded_image,
    ppe_label,
    summarize_result,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs" / "model.yaml"


def build_analyzer(
    configuration: ModelConfiguration,
    *,
    confidence_threshold: float | None = None,
    image_size: int | None = None,
) -> ImageAnalyzer:
    """Construye el flujo de analisis con los parametros validados."""
    detector = YoloPpeDetector(
        configuration.model_path,
        configuration.class_mapping,
        device=configuration.device,
        confidence_threshold=(
            configuration.confidence_threshold
            if confidence_threshold is None
            else confidence_threshold
        ),
        iou_threshold=configuration.iou_threshold,
        image_size=configuration.image_size if image_size is None else image_size,
    )
    return ImageAnalyzer(detector)


def build_person_analyzer(
    configuration: ModelConfiguration,
    *,
    confidence_threshold: float,
    image_size: int,
) -> PersonPpeAnalyzer:
    """Construye los detectores COCO y EPP usados en el flujo por persona."""
    if configuration.person_model_path is None:
        raise FileNotFoundError("No se configuro el modelo local de personas")

    person_detector = YoloPersonDetector(
        configuration.person_model_path,
        device=configuration.device,
        confidence_threshold=configuration.person_confidence_threshold,
        iou_threshold=configuration.person_iou_threshold,
        image_size=image_size,
        maximum_detections=configuration.maximum_people,
    )
    ppe_detector = YoloPpeDetector(
        configuration.model_path,
        configuration.class_mapping,
        device=configuration.device,
        confidence_threshold=confidence_threshold,
        iou_threshold=configuration.iou_threshold,
        image_size=image_size,
    )
    return PersonPpeAnalyzer(
        person_detector,
        ppe_detector,
        frozenset(configuration.class_mapping.values()),
    )


def _analyzer_key(
    configuration: ModelConfiguration,
    confidence_threshold: float,
    image_size: int,
) -> tuple[object, ...]:
    return (
        str(configuration.model_path),
        tuple(
            sorted((key, value.value) for key, value in configuration.class_mapping.items())
        ),
        confidence_threshold,
        configuration.iou_threshold,
        image_size,
    )


def _person_analyzer_key(
    configuration: ModelConfiguration,
    confidence_threshold: float,
    image_size: int,
) -> tuple[object, ...]:
    return (
        "individual",
        str(configuration.person_model_path),
        str(configuration.model_path),
        confidence_threshold,
        configuration.person_confidence_threshold,
        configuration.person_iou_threshold,
        configuration.iou_threshold,
        image_size,
        configuration.maximum_people,
    )


def _inject_styles(st) -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #07111b; color: #e7f0f5; }
        [data-testid="stHeader"] { background: transparent; }
        .block-container { max-width: 1180px; padding-top: 2.5rem; }
        .labguard-kicker {
            color: #5de0bd; font-size: .78rem; font-weight: 700;
            letter-spacing: .18em; text-transform: uppercase;
        }
        .labguard-title {
            color: #f2f8fa; font-size: clamp(2.5rem, 7vw, 5.5rem);
            font-weight: 800; letter-spacing: -.06em; line-height: .95;
            margin: .35rem 0 1rem;
        }
        .labguard-lead { color: #9bb0bc; font-size: 1.08rem; max-width: 700px; }
        .labguard-rule {
            height: 1px; margin: 1.8rem 0; border: 0;
            background: linear-gradient(90deg, #5de0bd, #183545 55%, transparent);
        }
        .labguard-card {
            background: #0d1c28; border: 1px solid #183545; border-radius: 14px;
            min-height: 132px; padding: 1.1rem 1.2rem;
        }
        .labguard-card h4 { color: #82ead0; font-size: .82rem; letter-spacing: .08em; }
        .labguard-card p { color: #e7f0f5; margin: .55rem 0 0; }
        [data-testid="stFileUploader"] {
            background: #0d1c28; border: 1px solid #234656; border-radius: 16px;
            padding: .6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _items_text(items) -> str:
    labels = [ppe_label(item) for item in items]
    return ", ".join(labels) if labels else "Ninguno"


def _status_text(status: EvaluationStatus) -> str:
    return {
        EvaluationStatus.DETECTED: "Detectado",
        EvaluationStatus.UNDETERMINED: "No determinado",
        EvaluationStatus.NOT_EVALUABLE: "No evaluable",
    }[status]


def _render_person_results(st, result: PersonAnalysisResult) -> None:
    st.subheader("Resultado individual")
    if not result.people:
        st.warning("No se detectaron personas con el modelo COCO en esta fotografia.")
        return

    st.metric("Personas detectadas", len(result.people))
    st.caption(
        "Los identificadores son temporales y se asignan de izquierda a derecha en cada fotografia."
    )
    for analysis in result.people:
        with st.container(border=True):
            st.markdown(
                f"### {analysis.person.person_id} · persona {analysis.person.confidence:.1%}"
            )
            status_columns = st.columns(len(analysis.statuses))
            for column, item_status in zip(
                status_columns,
                analysis.statuses,
                strict=True,
            ):
                column.caption(ppe_label(item_status.item))
                column.write(f"**{_status_text(item_status.status)}**")

            if analysis.detections:
                for detection in analysis.detections:
                    st.write(
                        f"{ppe_label(detection.item)}: {detection.confidence:.1%}"
                    )
            else:
                st.caption(
                    "No se asignaron detecciones EPP a esta persona; el resultado queda "
                    "como No determinado para las clases evaluables."
                )


def run() -> None:
    """Ejecuta la primera experiencia funcional de analisis de imagenes."""
    try:
        import streamlit as st
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Falta Streamlit. Instala las dependencias del proyecto para iniciar la interfaz."
        ) from error

    st.set_page_config(page_title="LabGuard AI", layout="wide")
    _inject_styles(st)

    st.markdown(
        '<div class="labguard-kicker">Vision artificial para laboratorios</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="labguard-title">LabGuard AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="labguard-lead">Analiza una fotografia y revisa los elementos de '
        "proteccion personal visibles. El resultado es orientativo y requiere "
        "supervision humana.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="labguard-rule">', unsafe_allow_html=True)

    try:
        configuration = load_model_configuration(MODEL_CONFIG_PATH, project_root=PROJECT_ROOT)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        st.error(f"No se pudo cargar la configuracion: {error}")
        return

    st.info(
        "Alcance actual: el modelo reconoce guantes, gafas de seguridad y tapabocas. "
        "La bata y el gorro todavia no son evaluables y no se declara incumplimiento por su ausencia."
    )

    st.subheader("Ajustes de inferencia")
    confidence_column, size_column = st.columns(2)
    default_confidence = min(max(configuration.confidence_threshold, 0.10), 0.60)
    with confidence_column:
        confidence_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.10,
            max_value=0.60,
            value=float(default_confidence),
            step=0.01,
            help="Un valor menor recupera mas candidatos, pero puede aumentar falsos positivos.",
        )
    with size_column:
        image_size = st.slider(
            "Tamano de inferencia",
            min_value=512,
            max_value=640,
            value=min(max(configuration.image_size, 512), 640),
            step=32,
            help="Una imagen mayor puede ayudar con EPP pequenos, a costa de mas tiempo en CPU.",
        )
    warnings_enabled = st.toggle(
        "Mostrar advertencias visuales",
        value=False,
        help=(
            "Solo se dibujan cuando existe una boca localizada de forma fiable y evidencia "
            "explicita de ausencia de tapabocas."
        ),
    )
    if warnings_enabled:
        st.caption(
            "Advertencias activadas. El pipeline actual aun no localiza rostros, por lo que no "
            "se inventaran posiciones ni se mostraran iconos de ausencia."
        )
    individual_analysis_enabled = st.toggle(
        "Analisis individual por persona",
        value=False,
        help=(
            "Detecta personas con yolov8n.pt y ejecuta el modelo EPP dentro de cada recorte. "
            "Puede ser mas lento en CPU."
        ),
    )
    st.caption(
        f"Procesamiento en CPU | confianza {confidence_threshold:.2f} | imagen {image_size}px"
    )

    uploaded_file = st.file_uploader(
        "Carga una fotografia del laboratorio",
        type=["jpg", "jpeg", "png"],
        help="Formatos admitidos: JPG, JPEG y PNG.",
    )
    if uploaded_file is None:
        st.caption("La imagen se procesa localmente cuando seleccionas un archivo.")
        return

    try:
        original = decode_uploaded_image(uploaded_file.getvalue())
    except ValueError as error:
        st.error(f"No se pudo abrir la imagen: {error}")
        return

    try:
        result = None
        person_result = None
        with st.spinner("Analizando elementos de proteccion personal..."):
            if individual_analysis_enabled:
                key = _person_analyzer_key(
                    configuration,
                    confidence_threshold,
                    image_size,
                )
                if st.session_state.get("labguard_person_analyzer_key") != key:
                    st.session_state["labguard_person_analyzer"] = build_person_analyzer(
                        configuration,
                        confidence_threshold=confidence_threshold,
                        image_size=image_size,
                    )
                    st.session_state["labguard_person_analyzer_key"] = key
                analyzer = st.session_state["labguard_person_analyzer"]
                person_result = analyzer.analyze(original)
                individual_detections = tuple(
                    detection
                    for analysis in person_result.people
                    for detection in analysis.detections
                )
                annotated = annotate_image(original, individual_detections)
                annotated = annotate_people(annotated, person_result.people)
            else:
                key = _analyzer_key(configuration, confidence_threshold, image_size)
                if st.session_state.get("labguard_analyzer_key") != key:
                    st.session_state["labguard_analyzer"] = build_analyzer(
                        configuration,
                        confidence_threshold=confidence_threshold,
                        image_size=image_size,
                    )
                    st.session_state["labguard_analyzer_key"] = key
                analyzer = st.session_state["labguard_analyzer"]
                result = analyzer.analyze(original)
                # No hay advertencias hasta integrar evidencia facial fiable por persona.
                warning_markers = ()
                annotated = annotate_image(
                    original,
                    result.detections,
                    warnings=warning_markers if warnings_enabled else (),
                )
    except FileNotFoundError as error:
        st.error(
            f"No se pudo cargar un modelo local: {error}. "
            "Verifica models/best.pt y yolov8n.pt antes de analizar imagenes."
        )
        return
    except Exception as error:
        st.error(f"El analisis no pudo completarse: {error}")
        return

    original_column, analyzed_column = st.columns(2)
    with original_column:
        st.subheader("Imagen original")
        st.image(original, channels="BGR", use_container_width=True)
    with analyzed_column:
        st.subheader("Imagen analizada")
        st.image(annotated, channels="BGR", use_container_width=True)

    st.markdown('<hr class="labguard-rule">', unsafe_allow_html=True)
    if person_result is not None:
        _render_person_results(st, person_result)
        st.caption(
            "No determinado significa que el EPP no pudo confirmarse para esa persona. "
            "No equivale a ausencia ni incumplimiento."
        )
        return

    st.subheader("Resultado del analisis")

    supported_items = tuple(
        item for item in PpeType if item in configuration.class_mapping.values()
    )
    counts = count_detections(result.detections)
    count_columns = st.columns(len(supported_items))
    for column, item in zip(count_columns, supported_items, strict=True):
        column.metric(ppe_label(item), counts.get(item, 0))

    st.info(
        "Cada recuadro representa una deteccion individual. El modelo actual no asocia EPP con "
        "personas concretas: detectar un guante no significa que todas las personas tengan guantes."
    )

    if result.detections:
        instance_counts = {item: 0 for item in supported_items}
        for detection in result.detections:
            instance_counts[detection.item] += 1
            label_column, confidence_column = st.columns([3, 2])
            label_column.write(
                f"**{ppe_label(detection.item)} #{instance_counts[detection.item]}**"
            )
            confidence_column.write(f"Confianza: **{detection.confidence:.1%}**")
            st.progress(min(max(float(detection.confidence), 0.0), 1.0))
    else:
        st.warning("No se reconocieron EPP compatibles con el modelo actual en esta imagen.")

    summary = summarize_result(result, supported_items)
    recognized_column, missing_column, unavailable_column = st.columns(3)
    with recognized_column:
        st.markdown(
            '<div class="labguard-card"><h4>RECONOCIDOS</h4>'
            f'<p>{_items_text(summary.recognized)}</p></div>',
            unsafe_allow_html=True,
        )
    with missing_column:
        st.markdown(
            '<div class="labguard-card"><h4>POSIBLEMENTE NO VISIBLES</h4>'
            f'<p>{_items_text(summary.possibly_missing)}</p></div>',
            unsafe_allow_html=True,
        )
    with unavailable_column:
        st.markdown(
            '<div class="labguard-card"><h4>NO EVALUABLES AUN</h4>'
            f'<p>{_items_text(summary.unavailable)}</p></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "Una ausencia significa que el elemento no fue visible para el modelo; no demuestra por si sola "
        "un incumplimiento. LabGuard AI no sustituye la inspeccion de seguridad."
    )
