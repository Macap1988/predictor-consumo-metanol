import unicodedata
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Predictor de consumo unitario de metanol",
    page_icon="⚗️",
    layout="wide"
)

@st.cache_resource
def cargar_bundle():
    return joblib.load("modelo_metanol_app.joblib")

def normalizar(texto):
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.lower().replace("_", " ").strip()

st.title("Predictor de consumo unitario de metanol")
st.caption("Simulador de escenarios operativos · Modelo XGBoost")

try:
    bundle = cargar_bundle()
except Exception as exc:
    st.error(
        "No se pudo cargar modelo_metanol_app.joblib. "
        "Verifique que el archivo esté en la raíz del repositorio y que "
        "se haya ejecutado la celda de exportación en Google Colab."
    )
    st.exception(exc)
    st.stop()

pipeline = bundle["pipeline"]
columnas = bundle["columnas"]
base = bundle["valores_base"]
rangos = bundle["rangos"]
controles = bundle["controles"]
metricas = bundle.get("metricas", {})

st.info(
    "Este simulador modifica ocho variables seleccionadas. Las demás variables "
    "del modelo permanecen fijadas en sus medianas históricas de entrenamiento. "
    "Por tanto, representa un escenario condicionado, no una lectura automática "
    "en tiempo real de la planta."
)

def restablecer_controles():
    for control in controles:
        col = control["columna"]
        if col in columnas:
            st.session_state[f"control_{col}"] = float(base[col])

with st.sidebar:
    st.header("Condiciones operativas")
    st.caption("Ajuste los controles dentro del rango histórico observado.")
    valores = {}
    for control in controles:
        col = control["columna"]
        if col not in columnas:
            continue
        minimo, maximo = rangos[col]
        inicial = float(min(max(base[col], minimo), maximo))
        clave = f"control_{col}"
        if clave not in st.session_state:
            st.session_state[clave] = inicial
        if minimo == maximo:
            valores[col] = st.number_input(
                control["etiqueta"], value=float(inicial), disabled=True
            )
        else:
            paso = max((maximo - minimo) / 200.0, 0.001)
            valores[col] = st.slider(
                control["etiqueta"],
                min_value=float(minimo),
                max_value=float(maximo),
                value=float(st.session_state[clave]),
                step=float(paso),
                key=clave,
                help=f"Rango histórico: {minimo:.3f} a {maximo:.3f}"
            )

    st.button(
        "Restablecer valores históricos",
        use_container_width=True,
        on_click=restablecer_controles
    )

fila1, fila2 = st.columns([1.3, 1])
with fila1:
    st.subheader("Resultado de la simulación")
    entrada = {c: float(base[c]) for c in columnas}
    entrada.update(valores)
    X = pd.DataFrame([entrada], columns=columnas)
    try:
        pred = float(pipeline.predict(X)[0])
        st.metric(
            "Consumo unitario predicho para t+1",
            f"{pred:,.2f} kg metanol / t biodiésel"
        )

        if pred <= 106.0:
            st.success(
                "Clasificación orientativa: EXCELENTE. "
                "Umbral configurado en el notebook (≤106)."
            )
        elif pred <= 112.0:
            st.warning(
                "Clasificación orientativa: NORMAL. "
                "Umbral configurado en el notebook (106–112)."
            )
        else:
            st.error(
                "Clasificación orientativa: ALERTA. "
                "Umbral configurado en el notebook (>112)."
            )

        st.caption(
            "Los umbrales son reglas incorporadas en el notebook original; "
            "no constituyen por sí solos límites operativos ni garantías de ahorro."
        )
        st.subheader("Variables aplicadas")
        resumen = []
        for control in controles:
            col = control["columna"]
            if col in valores:
                resumen.append({
                    "Variable": control["etiqueta"],
                    "Valor aplicado": valores[col]
                })
        st.dataframe(pd.DataFrame(resumen), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"No fue posible calcular la predicción: {exc}")

with fila2:
    st.subheader("Información del modelo")
    st.write(f"**Algoritmo:** XGBoost dentro de un Pipeline")
    st.write(f"**Variables de entrada:** {len(columnas)}")
    st.write(f"**Controles modificables:** {len(controles)}")
    if metricas:
        for nombre, valor in metricas.items():
            st.write(f"**{nombre}:** {valor}")
    st.divider()
    st.caption(
        "La predicción depende de la calidad, vigencia y representatividad "
        "de los datos usados para entrenar el modelo."
    )

st.divider()
st.caption(
    "Versión inicial de demostración. Antes de usarla para decisiones de planta, "
    "validar el modelo con una partición temporal independiente y datos recientes."
)
