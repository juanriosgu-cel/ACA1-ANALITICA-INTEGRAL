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
    st.write("Se estima el ingreso mensual a partir de edad, antigüedad en la empresa, abandono, rol de trabajo y departamento, siguiendo las variables definidas en el notebook de regresión.")

    # Misma selección de variables y partición del notebook de regresión
    reg_df = df[["Attrition", "Age", "MonthlyIncome", "YearsAtCompany", "JobRole", "Department"]].copy()
    reg_df = reg_df.rename(columns={"Attrition":"Abandono", "Age":"Edad", "MonthlyIncome":"IngresoMensual", "YearsAtCompany":"AñosEnCompañia", "JobRole":"RolDeTrabajo", "Department":"Departamento"})
    reg_encoded = pd.get_dummies(reg_df, columns=["Abandono", "RolDeTrabajo", "Departamento"], drop_first=True)
    yreg = reg_encoded["IngresoMensual"]
    Xreg = reg_encoded.drop(columns=["IngresoMensual"])
    Xtr, Xte, ytr, yte = train_test_split(Xreg, yreg, test_size=0.20, random_state=42)
    scaler_reg = StandardScaler()
    Xtr_scaled = scaler_reg.fit_transform(Xtr)
    Xte_scaled = scaler_reg.transform(Xte)

    with st.expander("Parámetros de regularización (opcionales)"):
        alpha_ridge = st.slider("Alpha de Ridge", 0.01, 10.0, 1.0, 0.01, help="Valor predeterminado 1.0, como en el notebook. Al aumentar alpha, aumenta la penalización L2.")
        alpha_lasso = st.slider("Alpha de Lasso", 0.01, 2.0, 0.1, 0.01, help="Valor predeterminado 0.1, como en el notebook. Al aumentar alpha, aumenta la penalización L1.")

    reg_models = {
        "Lineal": (LinearRegression(), Xtr, Xte, None),
        "Ridge": (Ridge(alpha=alpha_ridge), Xtr_scaled, Xte_scaled, scaler_reg),
        "Lasso": (Lasso(alpha=alpha_lasso, max_iter=10000), Xtr_scaled, Xte_scaled, scaler_reg),
    }
    rows, fitted_reg, pred_reg = [], {}, {}
    for name, (model, train_x, test_x, scaler_used) in reg_models.items():
        model.fit(train_x, ytr)
        pred = model.predict(test_x)
        fitted_reg[name] = (model, scaler_used)
        pred_reg[name] = pred
        rows.append({"Modelo":name, "R²":r2_score(yte, pred), "RMSE":np.sqrt(mean_squared_error(yte,pred)), "MAE":mean_absolute_error(yte,pred), "MSE":mean_squared_error(yte,pred)})
    met_reg = pd.DataFrame(rows).sort_values("RMSE")
    best_reg = met_reg.iloc[0]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mejor modelo por RMSE", best_reg["Modelo"])
    c2.metric("R² del mejor modelo", f"{best_reg['R²']:.4f}")
    c3.metric("RMSE", f"${best_reg['RMSE']:,.0f}")
    c4.metric("MAE", f"${best_reg['MAE']:,.0f}")
    st.dataframe(met_reg.style.format({"R²":"{:.4f}","RMSE":"{:,.2f}","MAE":"{:,.2f}","MSE":"{:,.0f}"}), use_container_width=True)

    st.markdown("#### Comparación gráfica de modelos")
    met_long = met_reg.melt(id_vars="Modelo", value_vars=["R²","RMSE","MAE"], var_name="Métrica", value_name="Valor")
    # Gráficas separadas para no mezclar escalas incompatibles
    g1,g2 = st.columns(2)
    with g1:
        st.plotly_chart(px.bar(met_reg.sort_values("R²",ascending=False), x="Modelo", y="R²", title="R²: mayor es mejor", text_auto=".4f"), use_container_width=True)
    with g2:
        err_long = met_reg.melt(id_vars="Modelo", value_vars=["RMSE","MAE"], var_name="Métrica de error", value_name="Valor")
        st.plotly_chart(px.bar(err_long, x="Modelo", y="Valor", color="Métrica de error", barmode="group", title="Errores de predicción: menor es mejor"), use_container_width=True)

    st.markdown("#### Interpretación comparativa")
    st.write(f"**{best_reg['Modelo']}** obtiene el menor RMSE con los parámetros actuales. Su R² ({best_reg['R²']:.3f}) indica que explica aproximadamente el {best_reg['R²']:.1%} de la variabilidad del ingreso mensual en el conjunto de prueba. El RMSE es de ${best_reg['RMSE']:,.0f} y el MAE de ${best_reg['MAE']:,.0f}.")
    diff = (met_reg["RMSE"].max()-met_reg["RMSE"].min())
    st.write(f"La diferencia entre el RMSE más alto y el más bajo es de ${diff:,.2f}. Una diferencia pequeña significa que los modelos tienen un desempeño predictivo muy parecido; por eso, la selección debe considerar también la simplicidad y la estabilidad.")

    st.markdown("#### Valores reales frente a estimados")
    model_plot = st.selectbox("Seleccione el modelo para explorar sus predicciones", ["Ridge","Lineal","Lasso"], index=0, key="reg_model_plot")
    real_est = pd.DataFrame({"Ingreso real":yte.to_numpy(), "Ingreso estimado":pred_reg[model_plot]})
    fig_real = px.scatter(real_est, x="Ingreso real", y="Ingreso estimado", opacity=0.62, title=f"Ingreso real frente a estimado — {model_plot}")
    low = min(real_est.min().min(), real_est.max().min())
    high = max(real_est.max().max(), real_est.max().max())
    fig_real.add_trace(go.Scatter(x=[real_est["Ingreso real"].min(), real_est["Ingreso real"].max()], y=[real_est["Ingreso real"].min(), real_est["Ingreso real"].max()], mode="lines", name="Predicción ideal", line=dict(dash="dash")))
    st.plotly_chart(fig_real, use_container_width=True)
    st.caption("Los puntos cercanos a la diagonal representan estimaciones más próximas al ingreso observado; las distancias grandes indican errores mayores.")

    st.markdown("#### Coeficientes del modelo")
    coef_model = st.selectbox("Modelo para consultar coeficientes", ["Ridge","Lineal","Lasso"], index=0, key="reg_coef_model")
    coef_obj, _ = fitted_reg[coef_model]
    coef_df = pd.DataFrame({"Variable":Xreg.columns, "Coeficiente":coef_obj.coef_})
    coef_df["Magnitud absoluta"] = coef_df["Coeficiente"].abs()
    coef_df = coef_df.sort_values("Magnitud absoluta", ascending=False).drop(columns="Magnitud absoluta")
    st.dataframe(coef_df.head(15).style.format({"Coeficiente":"{:,.2f}"}), use_container_width=True)
    st.write("Los coeficientes positivos se asocian con un ingreso estimado mayor y los negativos con uno menor, manteniendo constantes las demás variables. En Ridge y Lasso se trabaja con predictores estandarizados, por lo que sus coeficientes no se interpretan directamente en las unidades originales de cada variable. Estas asociaciones no demuestran causalidad.")

    st.markdown("#### Estimar el ingreso para un perfil ingresado")
    st.write("Modifique las características y seleccione el modelo. La estimación es exploratoria y no representa una determinación salarial individual.")
    with st.form("form_reg_pred"): 
        a,b,c = st.columns(3)
        age_input = a.slider("Edad", int(df["Age"].min()), int(df["Age"].max()), 35)
        tenure_input = b.slider("Años en la empresa", int(df["YearsAtCompany"].min()), int(df["YearsAtCompany"].max()), 5)
        attr_input = c.selectbox("Condición de abandono", sorted(df["Attrition"].unique()), index=0)
        d,e = st.columns(2)
        role_input = d.selectbox("Rol de trabajo", sorted(df["JobRole"].unique()))
        dept_input = e.selectbox("Departamento", sorted(df["Department"].unique()))
        model_predict_name = st.selectbox("Modelo para estimar", ["Ridge","Lineal","Lasso"], index=0, key="reg_individual_model")
        submit_reg = st.form_submit_button("Estimar ingreso mensual")
    if submit_reg:
        raw_row = pd.DataFrame([{"Abandono":attr_input,"Edad":age_input,"AñosEnCompañia":tenure_input,"RolDeTrabajo":role_input,"Departamento":dept_input}])
        encoded_row = pd.get_dummies(raw_row, columns=["Abandono","RolDeTrabajo","Departamento"], drop_first=True)
        encoded_row = encoded_row.reindex(columns=Xreg.columns, fill_value=0)
        model_obj, scaler_obj = fitted_reg[model_predict_name]
        row_for_model = scaler_obj.transform(encoded_row) if scaler_obj is not None else encoded_row
        estimate = float(model_obj.predict(row_for_model)[0])
        st.metric("Ingreso mensual estimado", f"${estimate:,.0f}")
        st.caption(f"Estimación calculada con el modelo {model_predict_name}. El resultado depende de las variables incluidas y del conjunto de datos utilizado para entrenar.")

elif section == "Segmentación de empleados":
    st.subheader("Segmentación interactiva de empleados")
    st.write("Seleccione las variables y ajuste los parámetros para recalcular los grupos. Se conservan las variables centrales del notebook como selección predeterminada.")
    feature_options = {"Edad":"Age", "Ingreso mensual":"MonthlyIncome", "Años de experiencia total":"TotalWorkingYears", "Antigüedad en la empresa":"YearsAtCompany", "Distancia al hogar":"DistanceFromHome", "Nivel del cargo":"JobLevel", "Número de empresas trabajadas":"NumCompaniesWorked"}
    selected_labels = st.multiselect("Variables utilizadas para segmentar", list(feature_options.keys()), default=["Edad","Ingreso mensual","Años de experiencia total","Antigüedad en la empresa"], help="Las variables seleccionadas se estandarizan antes del agrupamiento.")
    if len(selected_labels) < 2:
        st.warning("Seleccione al menos dos variables para realizar la segmentación.")
        st.stop()
    selected_cols = [feature_options[x] for x in selected_labels]
    Xseg = df[selected_cols].copy()
    Xseg.columns = selected_labels
    scaler_seg = StandardScaler()
    Z = scaler_seg.fit_transform(Xseg)

    algo = st.selectbox("Método de segmentación", ["K-means","Jerárquico (Ward)","DBSCAN"], index=0)
    if algo == "K-means":
        k = st.slider("Número de grupos (K)", 2, 9, 5, help="Ajuste K para observar cómo cambia la composición y separación de los grupos.")
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
        noise = 0
    elif algo == "Jerárquico (Ward)":
        k = st.slider("Número de grupos", 2, 9, 5)
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
        noise = 0
    else:
        eps = st.slider("eps (radio de vecindad)", 0.1, 2.0, 0.5, 0.05, help="Radio de vecindad: valores mayores suelen conectar más observaciones.")
        min_samples = st.slider("Mínimo de observaciones por vecindad", 3, 20, 5, help="Cantidad mínima de puntos para formar una región densa.")
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
        noise = int((labels == -1).sum())

    cluster_ids = set(labels) - {-1}
    non_noise = labels != -1
    if len(cluster_ids) >= 2 and non_noise.sum() > len(cluster_ids):
        # Para DBSCAN se calcula Silhouette sobre observaciones que sí fueron asignadas a un grupo.
        score = silhouette_score(Z[non_noise], labels[non_noise])
    else:
        score = np.nan
    out = Xseg.copy()
    out["Grupo"] = labels.astype(str)
    st.markdown("#### Resultado de la segmentación")
    m1,m2,m3 = st.columns(3)
    m1.metric("Grupos identificados", len(cluster_ids))
    m2.metric("Silhouette", f"{score:.4f}" if np.isfinite(score) else "No calculable")
    m3.metric("Registros de ruido", f"{noise:,}" if algo == "DBSCAN" else "No aplica")

    # PCA se recalcula con las variables elegidas
    pca_seg = PCA(n_components=2, random_state=42)
    coords = pca_seg.fit_transform(Z)
    plot_seg = pd.DataFrame({"Componente principal 1":coords[:,0], "Componente principal 2":coords[:,1], "Grupo":labels.astype(str)})
    st.plotly_chart(px.scatter(plot_seg, x="Componente principal 1", y="Componente principal 2", color="Grupo", title=f"Visualización PCA — {algo}", opacity=.7), use_container_width=True)
    st.caption(f"Los dos componentes explican conjuntamente {pca_seg.explained_variance_ratio_.sum():.1%} de la varianza estandarizada. PCA facilita la visualización, pero la proyección 2D puede ocultar separaciones presentes en más dimensiones.")

    st.markdown("#### Perfil de los grupos")
    profile = out[out["Grupo"] != "-1"].groupby("Grupo")[selected_labels].agg(["count","mean"]).round(2)
    st.dataframe(profile, use_container_width=True)
    counts = out["Grupo"].value_counts().sort_index().rename_axis("Grupo").reset_index(name="Empleados")
    st.plotly_chart(px.bar(counts, x="Grupo", y="Empleados", color="Grupo", title="Cantidad de empleados por grupo"), use_container_width=True)
    if np.isfinite(score):
        if score < 0.25:
            quality = "La cohesión y separación observadas son débiles."
        elif score < 0.50:
            quality = "La cohesión y separación son moderadas; puede existir solapamiento entre grupos."
        else:
            quality = "La separación entre grupos es relativamente marcada según Silhouette."
        st.write(f"**Interpretación:** {quality} El índice Silhouette se acerca a 1 cuando los grupos están más compactos y separados; valores cercanos a 0 indican solapamiento y valores negativos pueden indicar asignaciones poco adecuadas.")
    else:
        st.write("**Interpretación:** no fue posible calcular Silhouette de forma válida con esta configuración; revise los parámetros para obtener al menos dos grupos con suficientes observaciones asignadas.")
    if algo == "DBSCAN":
        st.write(f"DBSCAN identificó {len(cluster_ids)} grupos y {noise} observaciones de ruido ({noise/len(df):.1%}). El ruido corresponde a registros que no cumplen los criterios de densidad; no implica automáticamente que sean errores.")
    else:
        st.write(f"{algo} asignó los registros a {len(cluster_ids)} grupos. Compare los promedios para describir los perfiles. Las etiquetas de grupo son arbitrarias y no representan por sí mismas nivel de desempeño ni riesgo laboral.")

    st.markdown("#### Comparación de métodos con las mismas variables")
    comparison = []
    km_labels = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(Z)
    hier_labels = AgglomerativeClustering(n_clusters=5, linkage="ward").fit_predict(Z)
    db_labels = DBSCAN(eps=0.5, min_samples=5).fit_predict(Z)
    for method, lab in [("K-means (K=5)",km_labels),("Jerárquico (K=5)",hier_labels),("DBSCAN (eps=0.5)",db_labels)]:
        ids = set(lab) - {-1}; mask = lab != -1
        sc = silhouette_score(Z[mask],lab[mask]) if len(ids)>=2 and mask.sum()>len(ids) else np.nan
        comparison.append({"Método":method,"Grupos":len(ids),"Silhouette":sc,"Ruido":int((lab==-1).sum())})
    comp_df = pd.DataFrame(comparison)
    st.dataframe(comp_df.style.format({"Silhouette":"{:.4f}"}), use_container_width=True)
    valid_comp = comp_df.dropna(subset=["Silhouette"])
    if not valid_comp.empty:
        best_method = valid_comp.loc[valid_comp["Silhouette"].idxmax()]
        st.write(f"**Mejor puntuación de Silhouette en esta comparación:** {best_method['Método']} ({best_method['Silhouette']:.4f}). Esta es una preferencia según ese criterio, no una garantía de que sea el método más útil para todos los propósitos.")
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
