# Analítica de talento humano — paquete de despliegue

## Archivos incluidos
- `app.py`: aplicación Streamlit integrada.
- `WA_Fn-UseC_-HR-Employee-Attrition.csv`: base IBM HR Analytics requerida por la aplicación.
- `requirements.txt`: dependencias para instalar en Streamlit Community Cloud.

## Despliegue en Streamlit Community Cloud
1. Descargue y descomprima este paquete.
2. En el repositorio GitHub NUEVO destinado a la aplicación integrada, cargue los tres archivos en la raíz del repositorio.
3. En Streamlit Community Cloud, cree una app vinculada a ese repositorio y seleccione `app.py` como archivo principal.
4. Espere a que termine la instalación de dependencias y el arranque.
5. Pruebe cada módulo desde la navegación lateral.

## Importante
- No sobrescriba el repositorio de la aplicación anterior de clasificación.
- Este paquete usa los modelos definidos en `app.py` y recalcula sus métricas al ejecutarse. Por tanto, los resultados de clasificación pueden diferir de los obtenidos en el notebook original. Si se requiere reproducir exactamente esos resultados, hay que recuperar los modelos y la partición originales antes de publicar.
- La compilación sintáctica de Python fue verificada. La ejecución real en Streamlit aún debe probarse después del despliegue.
