# Nuevo despliegue — Analítica de talento humano

Aplicación Streamlit independiente para explorar clasificación del abandono laboral, regresión del ingreso mensual y segmentación de empleados usando IBM HR Analytics.

## Archivos
- `app.py`: aplicación Streamlit con los tres módulos y sus interpretaciones.
- `requirements.txt`: dependencias.
- `WA_Fn-UseC_-HR-Employee-Attrition.csv`: base de datos utilizada.

## Publicación en Streamlit Community Cloud
1. Crea un repositorio nuevo en GitHub, distinto de `ACA2_Streamlit_Abandono`.
2. Sube los tres archivos de este paquete a la raíz del repositorio.
3. En Streamlit Community Cloud selecciona **Create app**.
4. Elige el repositorio nuevo, rama `main` y archivo principal `app.py`.
5. Pulsa **Deploy**. La URL resultante será nueva e independiente.

## Módulos
- **Clasificación del abandono:** Regresión Logística, KNN, Árbol de Decisión y Random Forest; métricas, matriz de confusión, curva ROC y predicción exploratoria.
- **Regresión del ingreso:** Lineal, Ridge y Lasso; R², RMSE, MAE, MSE, coeficientes y valores reales frente a estimados.
- **Segmentación:** K-means, jerárquico y DBSCAN; perfiles de grupos, Silhouette, ruido de DBSCAN, PCA y descarga de registros etiquetados.

## Nota metodológica
Esta versión vuelve a ajustar los modelos al abrir cada módulo, en lugar de requerir un archivo serializado de modelos. Por ello, las métricas pueden diferir de las de los notebooks si las particiones, variables, transformaciones o hiperparámetros no coinciden exactamente. Las interpretaciones se calculan a partir de los resultados mostrados en la aplicación. No se deben interpretar como evidencia causal ni utilizar como único fundamento para decisiones laborales.
