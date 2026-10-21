# Analítica de talento humano

Aplicación Streamlit con clasificación de abandono, regresión del ingreso y segmentación interactiva.

## Archivos
- `app.py`: aplicación integrada; conserva las secciones de clasificación de la aplicación mejorada y añade regresión y segmentación.
- `modelos_despliegue.joblib`: modelos y artefactos de clasificación para las gráficas y predicción individual.
- `WA_Fn-UseC_-HR-Employee-Attrition.csv`: base de datos IBM HR Analytics.
- `requirements.txt`: dependencias.

## Despliegue
En Streamlit Community Cloud use rama `main` y archivo principal `app.py`. Los resultados de clasificación del artefacto incluido se entrenaron con partición reproducible de la base; pueden diferir de la ejecución original de la ACA 1.
