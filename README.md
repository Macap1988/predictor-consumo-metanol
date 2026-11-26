# Despliegue del predictor de metanol con Streamlit

## Contenido
- `app.py`: interfaz web.
- `requirements.txt`: dependencias.
- `exportar_modelo_colab.py`: bloque para ejecutar en Colab y exportar el Pipeline con sus metadatos.

## Paso 1. Exportar el modelo desde Colab
1. Abre `Modelo_Final_C_U_Metanol.ipynb`.
2. Ejecuta las celdas necesarias hasta completar el entrenamiento final de `best_pipeline_pure`.
3. Asegúrate de que existan `X_train_ts_pure`, `X_test_ts_pure`, `y_test_ts` y `best_pipeline_pure`.
4. Ejecuta el contenido de `exportar_modelo_colab.py` en una celda nueva.
5. Descarga `modelo_metanol_app.joblib`.

**Importante:** el bloque exporta el pipeline y metadatos del flujo actual. No corrige la inconsistencia metodológica entre el flujo Bootstrap usado por el XGBoost final y la partición temporal estricta del análisis comparativo.

## Paso 2. Crear repositorio GitHub
Crea un repositorio y carga estos archivos:
- `app.py`
- `requirements.txt`
- `modelo_metanol_app.joblib`

No subas datos históricos, credenciales ni información confidencial de planta. El archivo del modelo también puede contener información sensible derivada del entrenamiento; revisa las políticas de tu organización antes de hacerlo público.

## Paso 3. Publicar
1. Ingresa a https://share.streamlit.io/
2. Conecta GitHub.
3. Selecciona `Create app`.
4. Elige el repositorio, rama `main` y archivo `app.py`.
5. Presiona `Deploy`.

## Limitaciones de esta versión
- Modifica ocho variables operativas; las demás quedan fijas en las medianas históricas.
- Los umbrales 106 y 112 kg/t proceden del código original y son orientativos.
- No se presenta intervalo de confianza: el cálculo `1.96 * desviación estándar de residuos` del notebook no demuestra por sí solo cobertura estadística del 95%.
- No incorpora aún el optimizador global; requiere validar restricciones, unidades, límites operativos y coherencia física.
- Para una puesta en producción se recomienda una validación temporal independiente y control de versiones compatibles de Python, scikit-learn y XGBoost.
