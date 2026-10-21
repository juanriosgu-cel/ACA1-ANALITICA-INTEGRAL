import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve,
    precision_recall_curve, r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Analítica de talento humano", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background: #f4f7fb; border: 1px solid #e1e8f0; padding: 14px; border-radius: 12px;}
h1,h2,h3 {color: #183b56;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

df = load_data()
st.title("Analítica de talento humano")
st.caption("Clasificación del abandono, regresión del ingreso y segmentación de empleados | IBM HR Analytics")
st.info("Aplicación académica para explorar resultados estadísticos. Las asociaciones observadas no demuestran causalidad ni deben usarse como única base para decisiones laborales.")

with st.sidebar:
    st.header("Navegación")
    section = st.radio("Seleccione el módulo", [
        "Resumen general", "Clasificación del abandono", "Regresión del ingreso", "Segmentación de empleados", "Metodología y limitaciones"
    ])
    st.divider()
    st.caption("Base de datos IBM HR Analytics")
    st.write(f"Registros: **{len(df):,}**")
    st.write(f"Variables: **{df.shape[1]}**")

if section == "Resumen general":
    st.subheader("Vista general de la base")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Empleados", f"{len(df):,}")
    c2.metric("Variables", df.shape[1])
    c3.metric("Abandono", f"{(df['Attrition']=='Yes').mean():.1%}")
    c4.metric("Ingreso mensual medio", f"${df['MonthlyIncome'].mean():,.0f}")
    st.write("La aplicación integra tres enfoques complementarios sobre la misma base: clasificación para estimar abandono, regresión para estimar ingreso mensual y agrupamiento no supervisado para explorar perfiles.")
    st.dataframe(df.head(10), use_container_width=True)
    st.subheader("Distribución del abandono")
    counts = df["Attrition"].value_counts().rename_axis("Condición").reset_index(name="Empleados")
    st.plotly_chart(px.bar(counts, x="Condición", y="Empleados", color="Condición", title="Distribución de la variable Attrition"), use_container_width=True)
    st.warning("La clase de abandono es minoritaria. Por ello, Accuracy por sí sola puede dar una impresión demasiado favorable del desempeño de un clasificador.")

elif section == "Clasificación del abandono":
    st.subheader("Clasificación del abandono laboral")
    st.write("Se comparan cuatro clasificadores usando una partición de entrenamiento/prueba reproducible. Las métricas se calculan en la ejecución de la aplicación.")
    target = "Attrition"
    X = df.drop(columns=[target, "EmployeeNumber", "Over18", "StandardHours", "EmployeeCount"], errors="ignore")
    y = (df[target] == "Yes").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cat = X.select_dtypes(include=["object"]).columns.tolist()
    num = X.select_dtypes(exclude=["object"]).columns.tolist()
    pre = ColumnTransformer([("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
                             ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    models = {
        "Regresión Logística": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42, min_samples_leaf=2)
    }
    rows, fitted = [], {}
    for name, model in models.items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        prob = pipe.predict_proba(Xte)[:,1]
        fitted[name] = (pipe,pred,prob)
        rows.append({"Modelo":name, "Accuracy":accuracy_score(yte,pred),
                     "Precision abandono":precision_score(yte,pred,zero_division=0),
                     "Recall abandono":recall_score(yte,pred,zero_division=0),
                     "F1 abandono":f1_score(yte,pred,zero_division=0),
                     "ROC-AUC":roc_auc_score(yte,prob), "PR-AUC":average_precision_score(yte,prob)})
    metrics = pd.DataFrame(rows).sort_values("F1 abandono", ascending=False)
    st.dataframe(metrics.style.format({c:"{:.3f}" for c in metrics.columns if c!="Modelo"}), use_container_width=True)
    best = metrics.iloc[0]
    st.markdown("#### Interpretación automática")
    st.write(f"En esta ejecución, **{best['Modelo']}** obtiene el F1 más alto para la clase de abandono ({best['F1 abandono']:.3f}). Su Recall es {best['Recall abandono']:.3f}; es decir, identifica aproximadamente el {best['Recall abandono']:.1%} de los abandonos reales del conjunto de prueba.")
    st.caption("Este ajuste de modelos se realiza al abrir la sección; los resultados pueden diferir de los notebooks si sus particiones, variables, hiperparámetros o preprocesamiento fueron distintos.")
    chosen = st.selectbox("Explorar matriz de confusión y curvas", list(fitted.keys()), index=list(fitted.keys()).index(best["Modelo"]))
    pipe,pred,prob = fitted[chosen]
    cm = confusion_matrix(yte,pred)
    fig = px.imshow(cm, text_auto=True, x=["Predijo no abandono","Predijo abandono"], y=["Real no abandono","Real abandono"], color_continuous_scale="Blues", title=f"Matriz de confusión: {chosen}")
    st.plotly_chart(fig, use_container_width=True)
    tn,fp,fn,tp = cm.ravel()
    st.write(f"**Lectura:** identificó correctamente {tp} casos de abandono y dejó pasar {fn} casos reales (falsos negativos). También clasificó como abandono {fp} empleados que no abandonaron (falsos positivos).")
    fpr,tpr,_ = roc_curve(yte,prob)
    st.plotly_chart(go.Figure(go.Scatter(x=fpr,y=tpr,mode="lines",name="ROC")).add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Referencia",line=dict(dash="dash"))).update_layout(title=f"Curva ROC: {chosen}",xaxis_title="Tasa de falsos positivos",yaxis_title="Tasa de verdaderos positivos"), use_container_width=True)
    st.markdown("#### Predicción exploratoria de un registro")
    with st.expander("Ingresar valores para estimar probabilidad"):
        vals = {}
        cols = st.columns(3)
        for i,col in enumerate(X.columns):
            if col in cat:
                opts = sorted(X[col].dropna().astype(str).unique().tolist())
                vals[col] = cols[i%3].selectbox(col, opts, key="cls_"+col)
            else:
                vals[col] = cols[i%3].number_input(col, value=float(X[col].median()), key="cls_"+col, step=1.0)
        if st.button("Estimar probabilidad", key="predict_cls"):
            p = pipe.predict_proba(pd.DataFrame([vals]))[0,1]
            st.metric("Probabilidad estimada de abandono", f"{p:.1%}")
            st.write("Esta probabilidad es una estimación estadística, no una certeza sobre la conducta futura de una persona.")

elif section == "Regresión del ingreso":
    st.subheader("Regresión lineal, Ridge y Lasso")
    st.write("La variable objetivo es el ingreso mensual. Se comparan tres modelos con la misma partición de entrenamiento y prueba.")
    data = df[["Attrition","Age","MonthlyIncome","YearsAtCompany","JobRole","Department"]].copy()
    data = data.rename(columns={"Attrition":"Abandono","Age":"Edad","MonthlyIncome":"IngresoMensual","YearsAtCompany":"AñosEnCompañia","JobRole":"RolDeTrabajo","Department":"Departamento"})
    dm = pd.get_dummies(data, columns=["Abandono","RolDeTrabajo","Departamento"], drop_first=True)
    yreg = dm["IngresoMensual"]
    Xreg = dm.drop(columns=["IngresoMensual"])
    Xtr, Xte, ytr, yte = train_test_split(Xreg, yreg, test_size=.2, random_state=42)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    regressors = {
        "Lineal": (LinearRegression(), Xtr, Xte),
        "Ridge": (Ridge(alpha=10), Xtr_s, Xte_s),
        "Lasso": (Lasso(alpha=10, max_iter=10000), Xtr_s, Xte_s)
    }
    rows, fitted = [], {}
    for name, (model, trainx, testx) in regressors.items():
        model.fit(trainx, ytr)
        pred = model.predict(testx)
        fitted[name] = {"model": model, "testx": testx, "pred": pred, "trainx": trainx}
        rows.append({"Modelo": name, "R²": r2_score(yte, pred),
                     "RMSE": np.sqrt(mean_squared_error(yte, pred)),
                     "MAE": mean_absolute_error(yte, pred),
                     "MSE": mean_squared_error(yte, pred)})
    met = pd.DataFrame(rows)
    best = met.sort_values(["RMSE", "MAE"], ascending=[True, True]).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Modelo recomendado por error", best["Modelo"])
    c2.metric("R² del modelo recomendado", f"{best['R²']:.4f}")
    c3.metric("RMSE del modelo recomendado", f"${best['RMSE']:,.0f}")
    st.markdown("#### Comparación de modelos")
    st.dataframe(met.style.format({"R²":"{:.4f}", "RMSE":"{:,.2f}", "MAE":"{:,.2f}", "MSE":"{:,.0f}"}), use_container_width=True, hide_index=True)
    met_long = met.melt(id_vars="Modelo", value_vars=["R²"], var_name="Métrica", value_name="Valor")
    st.plotly_chart(px.bar(met_long, x="Modelo", y="Valor", text="Valor", title="Comparación de R²", color="Modelo"), use_container_width=True)
    err_long = met.melt(id_vars="Modelo", value_vars=["RMSE", "MAE"], var_name="Métrica", value_name="Error")
    st.plotly_chart(px.bar(err_long, x="Modelo", y="Error", color="Métrica", barmode="group", title="Comparación de errores (menor es mejor)"), use_container_width=True)
    st.markdown("#### ¿Cuál es el mejor modelo?")
    st.write(f"Según el criterio principal de menor RMSE, **{best['Modelo']}** presenta el mejor resultado en esta ejecución. Su R² es {best['R²']:.3f}, equivalente a aproximadamente {best['R²']:.1%} de la variabilidad del ingreso mensual explicada en el conjunto de prueba.")
    st.write(f"El MAE es ${best['MAE']:,.0f}, por lo que el error absoluto medio de predicción es de esa magnitud. El RMSE es ${best['RMSE']:,.0f} y penaliza más los errores grandes.")
    st.caption("La selección se basa en una única partición reproducible. Si los resultados son muy próximos, la diferencia práctica puede ser pequeña. Deben verificarse frente a la partición y parámetros exactos del notebook antes de reportarlos como resultados definitivos.")
    st.markdown("#### Interpretación de coeficientes")
    show = st.selectbox("Seleccione el modelo para examinar sus coeficientes", list(fitted.keys()), key="reg_coef")
    model = fitted[show]["model"]
    coef = pd.DataFrame({"Variable": Xreg.columns, "Coeficiente": model.coef_})
    coef["Magnitud absoluta"] = coef["Coeficiente"].abs()
    coef = coef.sort_values("Magnitud absoluta", ascending=False)
    st.dataframe(coef.drop(columns=["Magnitud absoluta"]).head(20).style.format({"Coeficiente":"{:,.2f}"}), use_container_width=True, hide_index=True)
    st.write("Los coeficientes positivos se asocian con un ingreso estimado mayor y los negativos con uno menor, manteniendo constantes las demás variables. En Ridge y Lasso, los coeficientes se estiman con predictores estandarizados; por tanto, su magnitud no se interpreta directamente en las unidades originales de cada variable. Estas asociaciones no demuestran causalidad.")
    chosen = st.selectbox("Seleccione el modelo para comparar valores reales y estimados", list(fitted.keys()), key="regplot")
    preds = fitted[chosen]["pred"]
    compare = pd.DataFrame({"Ingreso real": yte.values, "Ingreso estimado": preds})
    fig_compare = px.scatter(compare, x="Ingreso real", y="Ingreso estimado", title=f"Ingreso real frente a estimado: {chosen}", opacity=.6)
    minv = min(compare["Ingreso real"].min(), compare["Ingreso estimado"].min())
    maxv = max(compare["Ingreso real"].max(), compare["Ingreso estimado"].max())
    fig_compare.add_trace(go.Scatter(x=[minv, maxv], y=[minv, maxv], mode="lines", name="Predicción ideal", line=dict(dash="dash")))
    st.plotly_chart(fig_compare, use_container_width=True)
    st.caption("Los puntos más cercanos a la diagonal ideal representan estimaciones más próximas al ingreso observado.")
elif section == "Segmentación de empleados":
    st.subheader("Segmentación no supervisada")
    st.write("Se comparan K-means, agrupamiento jerárquico (Ward) y DBSCAN usando las mismas cuatro variables estandarizadas: edad, ingreso mensual, experiencia total y antigüedad en la empresa.")
    features = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany"]
    Xseg = df[features].copy()
    Xseg.columns = ["Edad", "Ingreso", "Años_Experiencia", "Antiguedad"]
    Z = StandardScaler().fit_transform(Xseg)

    # Comparación inicial de referencia. Los parámetros se declaran para que la comparación sea reproducible.
    candidate_labels = {
        "K-means": KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(Z),
        "Jerárquico (Ward)": AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(Z),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=5).fit_predict(Z)
    }
    comparison_rows = []
    for name, labs in candidate_labels.items():
        ncl = len(set(labs) - {-1})
        valid = labs != -1
        nnoise = int((labs == -1).sum())
        sil = np.nan
        if ncl >= 2 and valid.sum() > ncl and len(set(labs[valid])) >= 2:
            try:
                sil = silhouette_score(Z[valid], labs[valid])
            except Exception:
                sil = np.nan
        comparison_rows.append({"Método": name, "Grupos": ncl, "Silhouette": sil,
                                "Ruido": nnoise, "% ruido": nnoise / len(df) * 100})
    comp = pd.DataFrame(comparison_rows)
    valid_comp = comp[np.isfinite(comp["Silhouette"])]
    best_name = valid_comp.sort_values("Silhouette", ascending=False).iloc[0]["Método"] if not valid_comp.empty else None
    st.markdown("#### Comparación de métodos")
    st.dataframe(comp.style.format({"Silhouette":"{:.4f}", "% ruido":"{:.2f}%"}), use_container_width=True, hide_index=True)
    chart = comp.dropna(subset=["Silhouette"])
    if not chart.empty:
        st.plotly_chart(px.bar(chart, x="Método", y="Silhouette", color="Método", text="Silhouette", title="Comparación del índice Silhouette (mayor es mejor)"), use_container_width=True)
    if best_name:
        best_row = comp[comp["Método"] == best_name].iloc[0]
        st.success(f"**Método con mayor Silhouette: {best_name}.** Silhouette = {best_row['Silhouette']:.4f}. Se considera la alternativa preferente según este criterio comparativo, no una garantía de que sea el mejor para todos los usos.")
        if best_row["Silhouette"] < 0.25:
            quality = "La separación y cohesión estimadas son débiles."
        elif best_row["Silhouette"] < 0.50:
            quality = "La estructura de los grupos es moderada y puede existir solapamiento entre perfiles."
        else:
            quality = "La separación de grupos es relativamente marcada según esta métrica."
        st.write(f"**Interpretación:** {quality} El índice oscila entre -1 y 1; valores más altos suelen indicar grupos más compactos y separados.")
    else:
        st.warning("No fue posible calcular Silhouette para los métodos comparados con estos parámetros. No se puede recomendar un método usando este criterio.")
    algo = st.selectbox("Explore los resultados de un método", ["K-means", "Jerárquico (Ward)", "DBSCAN"], index=["K-means", "Jerárquico (Ward)", "DBSCAN"].index(best_name) if best_name else 0)
    if algo == "K-means":
        k = st.slider("Número de grupos K", 2, 8, 5, key="seg_kmeans")
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
    elif algo == "Jerárquico (Ward)":
        k = st.slider("Número de grupos", 2, 8, 5, key="seg_ward")
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    else:
        eps = st.slider("eps (radio de vecindad)", 0.2, 2.0, 0.5, 0.1, key="seg_eps")
        min_samples = st.slider("Mínimo de observaciones", 3, 20, 5, key="seg_min")
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
    n_clusters = len(set(labels) - {-1})
    non_noise = labels != -1
    noise = int((labels == -1).sum())
    score = np.nan
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        try:
            score = silhouette_score(Z[non_noise], labels[non_noise])
        except Exception:
            pass
    out = Xseg.copy()
    out["Grupo"] = labels.astype(str)
    profile = out[out["Grupo"] != "-1"].groupby("Grupo")[Xseg.columns].agg(["count", "mean"]).round(2)
    st.markdown(f"#### Perfil de los grupos: {algo}")
    st.dataframe(profile, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Grupos identificados", n_clusters)
    c2.metric("Silhouette del método explorado", f"{score:.3f}" if np.isfinite(score) else "No calculable")
    c3.metric("Observaciones de ruido", f"{noise:,}" if algo == "DBSCAN" else "No aplica")
    if algo == "DBSCAN":
        st.write(f"**Interpretación de DBSCAN:** se identificaron {n_clusters} grupos y {noise} registros como ruido ({noise/len(df):.1%} de la base). No cumplen los criterios de densidad definidos por los parámetros; no significa automáticamente que sean errores.")
    else:
        st.write(f"**Interpretación de {algo}:** el método asignó los empleados a {n_clusters} grupos. Compare los promedios de edad, ingreso, experiencia y antigüedad para describir las diferencias observadas. Las etiquetas de grupo son arbitrarias y no implican por sí mismas nivel de desempeño o riesgo.")
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Z)
    plot = pd.DataFrame({"Componente 1": coords[:,0], "Componente 2": coords[:,1], "Grupo": labels.astype(str)})
    st.plotly_chart(px.scatter(plot, x="Componente 1", y="Componente 2", color="Grupo", title=f"Visualización PCA de la segmentación: {algo}", opacity=.65), use_container_width=True)
    st.caption(f"Los dos componentes PCA explican conjuntamente {pca.explained_variance_ratio_.sum():.1%} de la varianza estandarizada. La proyección bidimensional puede ocultar separaciones presentes en otras dimensiones.")
    st.download_button("Descargar empleados con grupo asignado", out.to_csv(index=False).encode("utf-8-sig"), "empleados_segmentados.csv", "text/csv")

else:
    st.subheader("Metodología y limitaciones")
    st.markdown("""
    **Clasificación:** se estima la clase `Attrition` (Yes/No). Recall, F1 y Precision se refieren a la clase positiva `Yes`; ROC-AUC y PR-AUC se calculan a partir de probabilidades.

    **Regresión:** se estima `MonthlyIncome` a partir de edad, antigüedad, abandono, rol y departamento. R² resume la proporción de variabilidad explicada; MAE y RMSE están expresados en unidades monetarias de la base.

    **Segmentación:** se agrupan empleados a partir de edad, ingreso, experiencia total y antigüedad, tras estandarizar las variables. Silhouette resume cohesión y separación; en DBSCAN los puntos con etiqueta -1 se consideran ruido.

    **Limitaciones:** los resultados dependen de esta base de datos y de las decisiones de preprocesamiento. No permiten establecer causalidad ni deben utilizarse como único criterio para decisiones que afecten a personas. Las predicciones y agrupaciones son exploratorias y requieren validación contextual antes de cualquier aplicación real.
    """)
    st.write("Base usada:", f"{len(df):,} filas y {df.shape[1]} columnas.")
    st.write("Valores faltantes totales:", int(df.isna().sum().sum()))
    st.write("Registros duplicados:", int(df.duplicated().sum()))
