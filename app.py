import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Predicción del abandono de empleados",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CARGA DE LOS MODELOS Y RESULTADOS
# ============================================================
@st.cache_resource
def cargar_datos():
    return joblib.load("modelos_despliegue.joblib")


try:
    data = cargar_datos()
except FileNotFoundError:
    st.error(
        "No se encontró el archivo modelos_despliegue.joblib. "
        "Verifique que se encuentre en la misma carpeta que app.py."
    )
    st.stop()

modelos = data["models"]
columnas_modelo = data["columns"]
metricas = pd.DataFrame(data["metrics"])
matrices = data["confusion_matrices"]
roc_data = data["roc_data"]
pr_data = data["pr_data"]
importancias = data.get("feature_importance", {})
coeficientes = data.get("coefficients", {})
tiempos = pd.DataFrame(data.get("optimization_times", []))
categoricas = data.get("categorical_values", {})
defaults = data.get("numeric_defaults", {})
mins = data.get("numeric_min", {})
maxs = data.get("numeric_max", {})
pca_data = pd.DataFrame(data.get("pca_data", []))
pca_variance = data.get("pca_variance", [])


# ============================================================
# ESTILO
# ============================================================
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }

    .small-note { color: #b8c0cc; font-size: 0.88rem; }
    [data-testid="stMetric"] { background: rgba(40, 55, 75, 0.55); border: 1px solid rgba(160,175,195,.22); padding: .8rem; border-radius: .65rem; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: inherit !important; }

    .section-card {
        padding: 0.9rem 1rem;
        border-radius: 0.65rem;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def obtener_fila_modelo(nombre):
    filas = metricas[metricas["Modelo"] == nombre]
    if filas.empty:
        return None
    return filas.iloc[0]


def preparar_entrada(datos):
    """Convierte los datos del formulario al esquema esperado por el modelo."""
    entrada = pd.DataFrame([datos])

    entrada_encoded = pd.get_dummies(
        entrada,
        drop_first=True,
        dtype=int
    )

    entrada_encoded = entrada_encoded.reindex(
        columns=columnas_modelo,
        fill_value=0
    )

    return entrada_encoded


def obtener_probabilidad_abandono(modelo, entrada_encoded):
    """Obtiene la probabilidad asociada a la clase 1/Abandonó."""
    probabilidades = modelo.predict_proba(entrada_encoded)[0]

    if hasattr(modelo, "classes_"):
        clases = list(modelo.classes_)
        if 1 in clases:
            return float(probabilidades[clases.index(1)])
        if "Abandonó" in clases:
            return float(probabilidades[clases.index("Abandonó")])

    # Compatibilidad con los modelos exportados originalmente.
    return float(probabilidades[-1])


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("Navegación")

pagina = st.sidebar.radio(
    "Seleccione una sección",
    [
        "Resumen",
        "Comparación de modelos",
        "Análisis del modelo",
        "Predicción individual",
        "Regresión del ingreso",
        "Segmentación de empleados",
        "Metodología"
    ]
)

st.sidebar.divider()

modelo_seleccionado = st.sidebar.selectbox(
    "Modelo para análisis",
    list(modelos.keys()),
    index=(
        list(modelos.keys()).index("Random Forest")
        if "Random Forest" in modelos else 0
    )
)

st.sidebar.caption(
    "El modelo final del estudio es Random Forest. "
    "La interfaz permite consultar también las alternativas evaluadas."
)


# ============================================================
# ENCABEZADO
# ============================================================
st.title("Analítica de talento humano")
st.caption(
    "Aplicación interactiva desarrollada a partir de los modelos "
    "evaluados en la ACA 1 – Caso 1."
)


# ============================================================
# RESUMEN
# ============================================================
if pagina == "Resumen":

    st.header("Resumen del estudio")

    mejor_recall = metricas.loc[metricas["Recall"].idxmax()]
    mejor_f1 = metricas.loc[metricas["F1"].idxmax()]
    mejor_auc = metricas.loc[metricas["AUC"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Modelos evaluados",
            len(modelos)
        )

    with c2:
        st.metric(
            "Mayor Recall",
            mejor_recall["Modelo"],
            f'{mejor_recall["Recall"]:.4f}'
        )

    with c3:
        st.metric(
            "Mayor F1",
            mejor_f1["Modelo"],
            f'{mejor_f1["F1"]:.4f}'
        )

    with c4:
        st.metric(
            "Mayor ROC-AUC",
            mejor_auc["Modelo"],
            f'{mejor_auc["AUC"]:.4f}'
        )

    st.info(
        "Para este estudio, la clase de interés es «Abandonó». "
        "Por ello, Recall y F1-score de esta clase tienen especial "
        "importancia para la selección del modelo."
    )

    st.subheader("Resultados principales")

    columnas_mostrar = [
        "Modelo", "Accuracy", "Precision",
        "Recall", "F1", "F1_Macro", "AUC", "PR_AUC"
    ]

    st.dataframe(
        metricas[columnas_mostrar].style.format({
            c: "{:.4f}" for c in columnas_mostrar
            if c != "Modelo"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Descargar tabla de resultados (CSV)",
        metricas[columnas_mostrar].to_csv(index=False).encode("utf-8"),
        "resultados_modelos_ACA1.csv",
        "text/csv"
    )

    st.subheader("Modelo seleccionado")

    if "Random Forest" in modelos:
        rf = obtener_fila_modelo("Random Forest")

        st.success(
            f"Random Forest fue seleccionado en el estudio por su "
            f"desempeño en la identificación de la clase «Abandonó»: "
            f"Recall = {rf['Recall']:.4f} y F1 = {rf['F1']:.4f}."
        )

    st.warning(
        "Los resultados son de apoyo analítico. Un caso clasificado "
        "como posible abandono no constituye una determinación causal "
        "ni una decisión automática sobre un empleado."
    )


# ============================================================
# COMPARACIÓN DE MODELOS
# ============================================================
elif pagina == "Comparación de modelos":

    st.header("Comparación de modelos")

    st.write(
        "La comparación se realiza sobre el conjunto de prueba utilizando "
        "las métricas calculadas durante el desarrollo de la ACA 1 – Caso 1."
    )

    tab1, tab2, tab3 = st.tabs([
        "📊 Métricas de desempeño",
        "⚠️ Errores de clasificación",
        "⏱️ Costo computacional"
    ])

    # --------------------------------------------------------
    # MÉTRICAS
    # --------------------------------------------------------
    with tab1:

        columnas = [
            "Modelo", "Accuracy", "Precision",
            "Recall", "F1", "F1_Macro",
            "AUC", "PR_AUC"
        ]

        st.dataframe(
            metricas[columnas].style.format({
                c: "{:.4f}" for c in columnas if c != "Modelo"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Comparación general")

        metricas_largas = metricas.melt(
            id_vars="Modelo",
            value_vars=[
                "Accuracy", "Precision", "Recall",
                "F1", "F1_Macro", "AUC", "PR_AUC"
            ],
            var_name="Métrica",
            value_name="Valor"
        )

        fig = px.bar(
            metricas_largas,
            x="Modelo",
            y="Valor",
            color="Métrica",
            barmode="group",
            title="Desempeño comparativo de los modelos"
        )
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        fig.update_layout(
            legend_title_text="Métrica",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Criterios principales para la clase «Abandonó»")

        metricas_interes = metricas[
            ["Modelo", "Recall", "F1"]
        ].melt(
            id_vars="Modelo",
            var_name="Métrica",
            value_name="Valor"
        )

        fig_interes = px.bar(
            metricas_interes,
            x="Modelo",
            y="Valor",
            color="Métrica",
            barmode="group",
            text="Valor",
            title="Recall y F1 de la clase «Abandonó»"
        )
        fig_interes.update_yaxes(range=[0, 1], tickformat=".0%")
        fig_interes.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )
        fig_interes.update_layout(
            legend_title_text="Métrica",
            hovermode="x unified"
        )
        st.plotly_chart(fig_interes, use_container_width=True)

        st.caption(
            "Estas dos métricas son especialmente relevantes porque el objetivo "
            "del estudio es identificar correctamente casos de abandono."
        )

        st.markdown("#### Capacidad de discriminación")

        auc_df = metricas[
            ["Modelo", "AUC", "PR_AUC"]
        ].melt(
            id_vars="Modelo",
            var_name="Métrica",
            value_name="Valor"
        )

        fig_auc = px.bar(
            auc_df,
            x="Modelo",
            y="Valor",
            color="Métrica",
            barmode="group",
            text="Valor",
            title="ROC-AUC y PR-AUC"
        )
        fig_auc.update_yaxes(range=[0, 1], tickformat=".0%")
        fig_auc.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )
        st.plotly_chart(fig_auc, use_container_width=True)

    # --------------------------------------------------------
    # ERRORES
    # --------------------------------------------------------
    with tab2:

        tabla_error = []

        for nombre in modelos:
            cm = np.array(matrices[nombre])
            tn, fp, fn, tp = cm.ravel()

            tabla_error.append({
                "Modelo": nombre,
                "Verdaderos positivos": int(tp),
                "Falsos negativos": int(fn),
                "Falsos positivos": int(fp),
                "Verdaderos negativos": int(tn)
            })

        df_error = pd.DataFrame(tabla_error)

        st.dataframe(
            df_error,
            use_container_width=True,
            hide_index=True
        )

        fig_error = px.bar(
            df_error,
            x="Modelo",
            y=[
                "Verdaderos positivos",
                "Falsos negativos"
            ],
            barmode="group",
            title="Verdaderos positivos y falsos negativos"
        )
        st.plotly_chart(fig_error, use_container_width=True)

        st.caption(
            "Los falsos negativos corresponden a casos reales de abandono "
            "que el modelo no identifica. Por esta razón son especialmente "
            "relevantes para este estudio."
        )

    # --------------------------------------------------------
    # COSTO COMPUTACIONAL
    # --------------------------------------------------------
    with tab3:

        if not tiempos.empty:

            tiempos_ordenados = tiempos.sort_values(
                "Tiempo_segundos"
            )

            fig_t = px.bar(
                tiempos_ordenados,
                x="Modelo",
                y="Tiempo_segundos",
                text="Tiempo_segundos",
                title="Tiempo de optimización de hiperparámetros"
            )

            fig_t.update_traces(
                texttemplate="%{text:.3f} s",
                textposition="outside"
            )

            st.plotly_chart(
                fig_t,
                use_container_width=True
            )

            st.dataframe(
                tiempos_ordenados.style.format(
                    {"Tiempo_segundos": "{:.3f}"}
                ),
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "El tiempo de optimización es un criterio complementario "
                "y no constituye el criterio principal de selección del modelo."
            )
        else:
            st.info(
                "No se encontró información de tiempos de optimización "
                "en el archivo de despliegue."
            )


# ============================================================
# ANÁLISIS DEL MODELO
# ============================================================
elif pagina == "Análisis del modelo":

    st.header(f"Análisis: {modelo_seleccionado}")

    fila = obtener_fila_modelo(modelo_seleccionado)

    if fila is None:
        st.error("No se encontraron métricas para el modelo seleccionado.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("Accuracy", f'{fila["Accuracy"]:.4f}')

    with c2:
        st.metric("Precision", f'{fila["Precision"]:.4f}')

    with c3:
        st.metric("Recall", f'{fila["Recall"]:.4f}')

    with c4:
        st.metric("F1", f'{fila["F1"]:.4f}')

    with c5:
        st.metric("ROC-AUC", f'{fila["AUC"]:.4f}')

    tab1, tab2, tab3, tab4 = st.tabs([
        "Matriz de confusión",
        "ROC y Precision-Recall",
        "Variables",
        "PCA"
    ])

    # --------------------------------------------------------
    # MATRIZ
    # --------------------------------------------------------
    with tab1:

        cm = np.array(matrices[modelo_seleccionado])

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=["Permanece", "Abandonó"],
            y=["Permanece", "Abandonó"],
            labels={
                "x": "Predicción",
                "y": "Valor real",
                "color": "Cantidad"
            },
            title=f"Matriz de confusión – {modelo_seleccionado}"
        )

        fig_cm.update_layout(
            xaxis_title="Predicción del modelo",
            yaxis_title="Valor real"
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

        tn, fp, fn, tp = cm.ravel()

        a, b, c, d = st.columns(4)

        with a:
            st.metric("Verdaderos positivos", int(tp))

        with b:
            st.metric("Falsos negativos", int(fn))

        with c:
            st.metric("Falsos positivos", int(fp))

        with d:
            st.metric("Verdaderos negativos", int(tn))

        st.warning(
            f"El modelo deja {int(fn)} casos reales de abandono "
            "clasificados como «No abandonó»."
        )

    # --------------------------------------------------------
    # ROC Y PR
    # --------------------------------------------------------
    with tab2:

        col_roc, col_pr = st.columns(2)

        with col_roc:

            roc = roc_data[modelo_seleccionado]

            fig_roc = go.Figure()

            fig_roc.add_trace(
                go.Scatter(
                    x=roc["fpr"],
                    y=roc["tpr"],
                    mode="lines",
                    name=f"AUC = {fila['AUC']:.4f}"
                )
            )

            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Referencia",
                    line=dict(dash="dash")
                )
            )

            fig_roc.update_layout(
                title="Curva ROC",
                xaxis_title="Tasa de falsos positivos",
                yaxis_title="Tasa de verdaderos positivos",
                yaxis=dict(range=[0, 1]),
                xaxis=dict(range=[0, 1])
            )

            st.plotly_chart(
                fig_roc,
                use_container_width=True
            )

        with col_pr:

            pr = pr_data[modelo_seleccionado]

            fig_pr = go.Figure()

            fig_pr.add_trace(
                go.Scatter(
                    x=pr["recall"],
                    y=pr["precision"],
                    mode="lines",
                    name=f"PR-AUC = {fila['PR_AUC']:.4f}"
                )
            )

            fig_pr.update_layout(
                title="Curva Precision-Recall",
                xaxis_title="Recall",
                yaxis_title="Precision",
                yaxis=dict(range=[0, 1]),
                xaxis=dict(range=[0, 1])
            )

            st.plotly_chart(
                fig_pr,
                use_container_width=True
            )

        st.caption(
            "La curva ROC muestra la capacidad de discriminación del modelo. "
            "La curva Precision-Recall resulta especialmente informativa cuando "
            "la identificación de la clase de interés es prioritaria."
        )

    # --------------------------------------------------------
    # VARIABLES
    # --------------------------------------------------------
    with tab3:

        if modelo_seleccionado in importancias:

            df_imp = pd.DataFrame(
                importancias[modelo_seleccionado]
            ).sort_values("Importancia", ascending=True)

            fig_imp = px.bar(
                df_imp,
                x="Importancia",
                y="Variable",
                orientation="h",
                title="Importancia de variables"
            )

            st.plotly_chart(
                fig_imp,
                use_container_width=True
            )

            st.caption(
                "La importancia predictiva indica la contribución relativa "
                "de las variables dentro del modelo y no implica causalidad."
            )

        elif modelo_seleccionado in coeficientes:

            df_coef = pd.DataFrame(
                coeficientes[modelo_seleccionado]
            )

            df_coef["Abs"] = df_coef["Coeficiente"].abs()
            df_coef = df_coef.sort_values("Abs", ascending=True)

            fig_coef = px.bar(
                df_coef,
                x="Coeficiente",
                y="Variable",
                orientation="h",
                title="Coeficientes de Regresión Logística"
            )

            st.plotly_chart(
                fig_coef,
                use_container_width=True
            )

            st.caption(
                "El signo del coeficiente indica dirección dentro del modelo; "
                "no debe interpretarse como causalidad."
            )

        else:
            if modelo_seleccionado == "KNN":
                st.info(
                    "KNN no dispone de coeficientes ni de una medida "
                    "feature_importances_ propia. Por ello, esta aplicación "
                    "no asigna una importancia de variables que no haya sido "
                    "calculada y exportada desde el notebook."
                )
                st.markdown(
                    "**Interpretación:** KNN realiza la clasificación a partir "
                    "de la proximidad entre observaciones en el espacio de "
                    "variables. Por esta razón, la lectura de variables "
                    "individuales requiere una técnica adicional, como "
                    "Permutation Importance, calculada sobre un conjunto de "
                    "datos de referencia."
                )
            else:
                st.info(
                    "Este modelo no cuenta con una medida directa de importancia "
                    "de variables exportada."
                )

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------
    with tab4:

        if not pca_data.empty:

            fig_pca = px.scatter_3d(
                pca_data,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Clase",
                symbol="Clase",
                title="Visualización 3D mediante PCA"
            )

            st.plotly_chart(
                fig_pca,
                use_container_width=True
            )

            if pca_variance:

                var = np.array(pca_variance)

                st.write(
                    f"Varianza explicada: PC1 = {var[0]:.4f}, "
                    f"PC2 = {var[1]:.4f}, PC3 = {var[2]:.4f}. "
                    f"Acumulada = {var.sum():.4f}."
                )

            st.caption(
                "El PCA se utiliza como recurso de visualización y no "
                "interviene en la selección del modelo."
            )

        else:
            st.info(
                "La información PCA aún no fue exportada desde el notebook."
            )


# ============================================================
# PREDICCIÓN INDIVIDUAL
# ============================================================
elif pagina == "Predicción individual":

    st.header("Predicción individual")

    st.write(
        "Diligencie las características del empleado. La aplicación "
        "utilizará el modelo seleccionado en el panel lateral."
    )

    st.info(
        "Esta predicción es una estimación del modelo. No representa "
        "una determinación definitiva sobre el comportamiento futuro "
        "de un empleado."
    )

    st.caption(
        f"**Modelo utilizado para la predicción:** {modelo_seleccionado}"
    )

    with st.form("formulario_prediccion"):

        datos = {}

        if categoricas:

            st.subheader("Información categórica")

            cols = st.columns(2)

            for i, variable in enumerate(categoricas):

                opciones = categoricas[variable]

                with cols[i % 2]:

                    datos[variable] = st.selectbox(
                        variable.replace("_", " "),
                        opciones
                    )

        numericas = [
            c for c in defaults
            if c not in categoricas
        ]

        if numericas:

            st.subheader("Información numérica")

            cols = st.columns(2)

            for i, variable in enumerate(numericas):

                minimo = float(mins[variable])
                maximo = float(maxs[variable])
                valor = float(defaults[variable])

                with cols[i % 2]:

                    if minimo == maximo:

                        datos[variable] = st.number_input(
                            variable.replace("_", " "),
                            value=valor,
                            disabled=True
                        )

                    else:

                        valor = min(
                            max(valor, minimo),
                            maximo
                        )

                        datos[variable] = st.number_input(
                            variable.replace("_", " "),
                            min_value=minimo,
                            max_value=maximo,
                            value=valor,
                            step=1.0
                        )

        ejecutar = st.form_submit_button(
            "Realizar predicción",
            type="primary"
        )

    if ejecutar:

        entrada_encoded = preparar_entrada(datos)

        modelo = modelos[modelo_seleccionado]

        prediccion = int(
            modelo.predict(entrada_encoded)[0]
        )

        probabilidad = obtener_probabilidad_abandono(
            modelo,
            entrada_encoded
        )

        st.divider()
        st.subheader("Resultado")

        if prediccion == 1:
            st.error("Resultado: Abandonó")
            resultado_texto = "Abandonó"
        else:
            st.success("Resultado: Permanece")
            resultado_texto = "Permanece"

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Probabilidad de abandono",
                f"{probabilidad:.2%}"
            )

        with c2:
            st.metric(
                "Probabilidad de permanencia",
                f"{1 - probabilidad:.2%}"
            )

        st.subheader("Interpretación del resultado")

        if prediccion == 1:
            st.write(
                f"El modelo **{modelo_seleccionado}** clasifica el caso como "
                f"**«Abandonó»**, con una probabilidad estimada de abandono "
                f"del **{probabilidad:.2%}**."
            )
        else:
            st.write(
                f"El modelo **{modelo_seleccionado}** clasifica el caso como "
                f"**«Permanece»**, con una probabilidad estimada de abandono "
                f"del **{probabilidad:.2%}** y una probabilidad de permanencia "
                f"del **{1 - probabilidad:.2%}**."
            )

        st.caption(
            "La clasificación corresponde al resultado del modelo sobre las "
            "características diligenciadas. La probabilidad estimada no "
            "representa una certeza ni una relación causal."
        )

        fig_prob = px.bar(
            x=["Abandonó", "Permanece"],
            y=[probabilidad, 1 - probabilidad],
            labels={
                "x": "",
                "y": "Probabilidad"
            },
            title="Probabilidades estimadas"
        )

        fig_prob.update_yaxes(
            range=[0, 1],
            tickformat=".0%"
        )
        fig_prob.update_traces(
            texttemplate="%{y:.1%}",
            textposition="outside"
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )


# ============================================================
# METODOLOGÍA
# ============================================================
elif pagina == "Regresión del ingreso":

    st.header("Regresión del ingreso mensual")
    st.write("Compare los modelos Lineal, Ridge y Lasso, explore sus errores y estime el ingreso mensual para un perfil ingresado. Los resultados son exploratorios.")
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    reg_df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    cols = ["Attrition", "Age", "MonthlyIncome", "YearsAtCompany", "JobRole", "Department"]
    reg_df = reg_df[cols].copy().rename(columns={"Attrition":"Abandono","Age":"Edad","MonthlyIncome":"IngresoMensual","YearsAtCompany":"AñosEnCompañia","JobRole":"RolDeTrabajo","Department":"Departamento"})
    encoded = pd.get_dummies(reg_df, columns=["Abandono","RolDeTrabajo","Departamento"], drop_first=True, dtype=int)
    yreg = encoded["IngresoMensual"]; Xreg = encoded.drop(columns="IngresoMensual")
    Xtr, Xte, ytr, yte = train_test_split(Xreg, yreg, test_size=.20, random_state=42)
    scaler = StandardScaler(); Xtrs = scaler.fit_transform(Xtr); Xtes = scaler.transform(Xte)
    with st.expander("Parámetros de regularización"):
        ar = st.slider("Alpha de Ridge", 0.01, 10.0, 1.0, 0.01)
        al = st.slider("Alpha de Lasso", 0.01, 2.0, 0.1, 0.01)
    reg_models = {"Lineal":(LinearRegression(),Xtr,Xte,None),"Ridge":(Ridge(alpha=ar),Xtrs,Xtes,scaler),"Lasso":(Lasso(alpha=al,max_iter=10000),Xtrs,Xtes,scaler)}
    regfit={}; rows=[]
    for nm,(mdl,xtrain,xtest,sc) in reg_models.items():
        mdl.fit(xtrain,ytr); pred=mdl.predict(xtest); regfit[nm]=(mdl,sc)
        rows.append({"Modelo":nm,"R²":r2_score(yte,pred),"MAE":mean_absolute_error(yte,pred),"RMSE":mean_squared_error(yte,pred)**.5,"MSE":mean_squared_error(yte,pred)})
    rmetrics=pd.DataFrame(rows).sort_values("RMSE")
    best=rmetrics.iloc[0]
    st.subheader("Comparación de modelos")
    st.dataframe(rmetrics.style.format({c:"{:.3f}" for c in ["R²","MAE","RMSE","MSE"]}),use_container_width=True,hide_index=True)
    a,b,c=st.columns(3); a.metric("Mejor modelo por RMSE",best["Modelo"]); b.metric("R²",f"{best['R²']:.3f}"); c.metric("RMSE",f"${best['RMSE']:,.0f}")
    st.success(f"Según el menor RMSE, {best['Modelo']} presenta el menor error de predicción en esta partición. R² = {best['R²']:.3f}; MAE = ${best['MAE']:,.0f}. Estas métricas comparan ajuste y error, no prueban causalidad.")
    long=rmetrics.melt(id_vars="Modelo",value_vars=["R²","MAE","RMSE"],var_name="Métrica",value_name="Valor")
    st.plotly_chart(px.bar(long,x="Modelo",y="Valor",color="Métrica",barmode="group",title="Comparación de métricas de regresión"),use_container_width=True)
    nmplot=st.selectbox("Modelo para explorar predicciones",["Ridge","Lineal","Lasso"],index=0,key="reg_plot")
    mdl,sc=regfit[nmplot]; testpred=mdl.predict(sc.transform(Xte) if sc else Xte)
    realpred=pd.DataFrame({"Ingreso real":yte.values,"Ingreso estimado":testpred})
    fig=px.scatter(realpred,x="Ingreso real",y="Ingreso estimado",title=f"Ingreso real frente a estimado: {nmplot}",opacity=.65)
    lo=min(realpred.min()); hi=max(realpred.max()); fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Predicción ideal",line=dict(dash="dash")))
    st.plotly_chart(fig,use_container_width=True)
    resid=yte.values-testpred
    st.plotly_chart(px.scatter(x=testpred,y=resid,labels={"x":"Ingreso estimado","y":"Residuo"},title=f"Residuos del modelo {nmplot}"),use_container_width=True)
    st.write("**Interpretación:** puntos próximos a la línea ideal representan estimaciones más cercanas al ingreso observado. Los residuos muestran la diferencia entre ingreso real y estimado; patrones sistemáticos pueden indicar errores de ajuste.")
    st.subheader("Coeficientes del modelo")
    coefnm=st.selectbox("Modelo para consultar coeficientes",["Lineal","Ridge","Lasso"],key="reg_coef")
    cm,cs=regfit[coefnm]; coef=cm.coef_
    cdf=pd.DataFrame({"Variable":Xreg.columns,"Coeficiente":coef}).assign(Abs=lambda z:z.Coeficiente.abs()).sort_values("Abs").tail(20)
    st.plotly_chart(px.bar(cdf,x="Coeficiente",y="Variable",orientation="h",title=f"Coeficientes principales: {coefnm}"),use_container_width=True)
    st.caption("En Ridge y Lasso los predictores están estandarizados. Los coeficientes describen asociaciones condicionadas al modelo y no implican causalidad.")
    st.subheader("Estimar ingreso para un perfil")
    with st.form("reg_individual"):
        a,b,c=st.columns(3); age=a.slider("Edad",18,60,35); tenure=b.slider("Años en la empresa",0,40,5); attr=c.selectbox("Condición de abandono",sorted(reg_df["Abandono"].unique()))
        d,e=st.columns(2); role=d.selectbox("Rol de trabajo",sorted(reg_df["RolDeTrabajo"].unique())); dept=e.selectbox("Departamento",sorted(reg_df["Departamento"].unique()))
        which=st.selectbox("Modelo para estimar ingreso",["Ridge","Lineal","Lasso"],index=0); submit=st.form_submit_button("Estimar ingreso mensual")
    if submit:
        raw=pd.DataFrame([{"Abandono":attr,"Edad":age,"AñosEnCompañia":tenure,"RolDeTrabajo":role,"Departamento":dept}])
        enc=pd.get_dummies(raw,columns=["Abandono","RolDeTrabajo","Departamento"],drop_first=True,dtype=int).reindex(columns=Xreg.columns,fill_value=0)
        mm,ss=regfit[which]; est=float(mm.predict(ss.transform(enc) if ss else enc)[0]); st.metric("Ingreso mensual estimado",f"${est:,.0f}"); st.caption(f"Estimación exploratoria calculada con {which}; no representa una determinación salarial individual.")

elif pagina == "Segmentación de empleados":

    st.header("Segmentación interactiva de empleados")
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    dseg=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    options={"Edad":"Age","Ingreso mensual":"MonthlyIncome","Experiencia total":"TotalWorkingYears","Antigüedad en la empresa":"YearsAtCompany","Distancia al hogar":"DistanceFromHome","Nivel del cargo":"JobLevel","Empresas anteriores":"NumCompaniesWorked","Satisfacción ambiental":"EnvironmentSatisfaction","Equilibrio vida-trabajo":"WorkLifeBalance"}
    selected=st.multiselect("Variables utilizadas para segmentar",list(options),default=["Edad","Ingreso mensual","Experiencia total","Antigüedad en la empresa"])
    if len(selected)<2:
        st.warning("Seleccione al menos dos variables para calcular la segmentación.")
    else:
        raw=dseg[[options[v] for v in selected]].copy(); raw.columns=selected
        scaler_s=StandardScaler(); Z=scaler_s.fit_transform(raw)
        algo=st.selectbox("Método de segmentación",["K-means","Jerárquico (Ward)","DBSCAN"],key="seg_algo")
        if algo=="K-means":
            k=st.slider("Número de grupos (K)",2,9,5); labels=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(Z)
        elif algo=="Jerárquico (Ward)":
            k=st.slider("Número de grupos",2,9,5,key="seg_hier_k"); labels=AgglomerativeClustering(n_clusters=k,linkage="ward").fit_predict(Z)
        else:
            eps=st.slider("eps (radio de vecindad)",0.1,2.0,0.5,0.05); mins=st.slider("Mínimo de observaciones",3,20,5); labels=DBSCAN(eps=eps,min_samples=mins).fit_predict(Z)
        ids=set(labels)-{-1}; mask=labels!=-1
        score=silhouette_score(Z[mask],labels[mask]) if len(ids)>=2 and mask.sum()>len(ids) else np.nan
        c1,c2,c3=st.columns(3); c1.metric("Grupos identificados",len(ids)); c2.metric("Silhouette",f"{score:.4f}" if np.isfinite(score) else "No calculable"); c3.metric("Ruido DBSCAN",f"{int((labels==-1).sum()):,}" if algo=="DBSCAN" else "No aplica")
        pca=PCA(n_components=2,random_state=42); coords=pca.fit_transform(Z)
        plot=pd.DataFrame({"Componente 1":coords[:,0],"Componente 2":coords[:,1],"Grupo":labels.astype(str)})
        st.plotly_chart(px.scatter(plot,x="Componente 1",y="Componente 2",color="Grupo",title=f"PCA de la segmentación: {algo}",opacity=.7),use_container_width=True)
        st.caption(f"Los dos componentes explican {pca.explained_variance_ratio_.sum():.1%} de la varianza estandarizada.")
        assigned=raw.copy(); assigned["Grupo"]=labels.astype(str)
        st.subheader("Tamaño de los grupos")
        counts=assigned["Grupo"].value_counts().sort_index().rename_axis("Grupo").reset_index(name="Empleados")
        st.plotly_chart(px.bar(counts,x="Grupo",y="Empleados",color="Grupo",title="Empleados por grupo"),use_container_width=True)
        st.subheader("Perfil de cada grupo")
        profile=assigned[assigned["Grupo"]!="-1"].groupby("Grupo")[selected].agg(["count","mean"]).round(2)
        st.dataframe(profile,use_container_width=True)
        if np.isfinite(score):
            quality="débil" if score<.25 else "moderada" if score<.5 else "relativamente marcada"
            st.write(f"**Interpretación:** la cohesión y separación son {quality} según Silhouette ({score:.3f}). Los grupos pueden solaparse; interprete los perfiles antes de asignarles significado.")
        else: st.warning("Silhouette no puede calcularse de forma válida para esta configuración; ajuste los parámetros.")
        st.download_button("Descargar empleados con grupo asignado",assigned.to_csv(index=False).encode("utf-8-sig"),"empleados_segmentados.csv","text/csv")

elif pagina == "Metodología":

    st.header("Metodología del modelamiento")

    st.write(
        "La aplicación presenta los resultados del proceso de modelamiento "
        "desarrollado para la ACA 1 – Caso 1 y permite consultar los modelos "
        "evaluados, sus métricas y sus resultados de predicción."
    )

    pasos = [
        (
            "1. Datos",
            "Preparación del conjunto de datos utilizado para el estudio "
            "del abandono de empleados."
        ),
        (
            "2. Preprocesamiento",
            "Transformación y preparación de las variables para que puedan "
            "ser utilizadas por los algoritmos de clasificación."
        ),
        (
            "3. Modelamiento",
            "Evaluación de Regresión Logística, KNN, Árbol de Decisión "
            "y Random Forest."
        ),
        (
            "4. Optimización",
            "Ajuste de los modelos mediante los procedimientos definidos "
            "en el desarrollo de la ACA 1 – Caso 1."
        ),
        (
            "5. Evaluación",
            "Comparación mediante Accuracy, Precision, Recall, F1, "
            "ROC-AUC y PR-AUC, además de la matriz de confusión."
        ),
        (
            "6. Selección",
            "Random Forest fue seleccionado considerando especialmente "
            "el desempeño en la identificación de la clase «Abandonó»."
        ),
        (
            "7. Despliegue",
            "Los modelos y resultados fueron integrados en una aplicación "
            "interactiva desarrollada con Streamlit."
        )
    ]

    for titulo, descripcion in pasos:
        with st.container(border=True):
            st.subheader(titulo)
            st.write(descripcion)

    st.subheader("Criterio de selección del modelo")

    if "Random Forest" in modelos:

        rf = obtener_fila_modelo("Random Forest")

        st.success(
            f"Random Forest presenta Recall = {rf['Recall']:.4f} y "
            f"F1 = {rf['F1']:.4f} para la clase «Abandonó». "
            "Estas métricas fueron consideradas prioritarias para el "
            "objetivo del estudio."
        )

    st.subheader("Interpretación")

    st.write(
        "Las métricas y visualizaciones deben interpretarse como evidencia "
        "del desempeño predictivo de los modelos sobre los datos evaluados. "
        "La importancia de variables y las predicciones no implican "
        "relaciones causales."
    )

    st.warning(
        "La aplicación tiene finalidad académica y de apoyo analítico. "
        "Una predicción de posible abandono no constituye una decisión "
        "automática sobre un empleado."
    )

    st.caption(
        "El despliegue reproduce los modelos y resultados exportados desde "
        "el notebook; no realiza un nuevo entrenamiento durante la consulta."
    )


# ============================================================
# PIE DE PÁGINA
# ============================================================
st.divider()

st.caption(
    "Proyecto académico – Especialización en Inteligencia Artificial. "
    "Los modelos corresponden a la ejecución documentada en el notebook de la ACA 1 – Caso 1."
)
