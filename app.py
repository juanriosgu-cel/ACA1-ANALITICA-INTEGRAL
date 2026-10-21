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
div[data-testid="stMetric"] {background: rgba(128,128,128,0.10); border: 1px solid rgba(128,128,128,0.25); padding: 14px; border-radius: 12px;}
h1,h2,h3 {letter-spacing: -0.02em;}
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
    best = metrics.iloc[0]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Modelos comparados", len(metrics))
    c2.metric("Mejor F1 abandono", best["Modelo"], f"{best['F1 abandono']:.3f}")
    c3.metric("Recall del mejor", f"{best['Recall abandono']:.3f}")
    c4.metric("ROC-AUC del mejor", f"{best['ROC-AUC']:.3f}")
    st.success(f"**Mejor modelo según F1 para la clase Abandonó: {best['Modelo']}**. El Recall es {best['Recall abandono']:.3f}, por lo que identifica aproximadamente el {best['Recall abandono']:.1%} de los abandonos reales en esta partición.")
    st.markdown("### Comparación de modelos")
    st.dataframe(metrics.style.format({c:"{:.4f}" for c in metrics.columns if c!="Modelo"}), use_container_width=True, hide_index=True)
    metric_long = metrics.melt(id_vars="Modelo", value_vars=["Accuracy","Precision abandono","Recall abandono","F1 abandono","ROC-AUC","PR-AUC"], var_name="Métrica", value_name="Valor")
    st.plotly_chart(px.bar(metric_long, x="Modelo", y="Valor", color="Métrica", barmode="group", title="Comparación completa de métricas por modelo", text_auto=".3f").update_yaxes(range=[0,1]), use_container_width=True)
    st.markdown("### Análisis detallado del modelo")
    chosen = st.selectbox("Seleccione el modelo que desea analizar", list(fitted.keys()), index=list(fitted.keys()).index(best["Modelo"]), key="class_model_select")
    pipe,pred,prob = fitted[chosen]
    tab1,tab2,tab3,tab4 = st.tabs(["Matriz de confusión", "Curvas ROC y Precision-Recall", "Variables", "PCA"])
    with tab1:
        cm = confusion_matrix(yte,pred)
        st.plotly_chart(px.imshow(cm, text_auto=True, x=["Predijo no abandono","Predijo abandono"], y=["Real no abandono","Real abandono"], color_continuous_scale="Blues", title=f"Matriz de confusión: {chosen}"), use_container_width=True)
        tn,fp,fn,tp = cm.ravel()
        a,b,c,d = st.columns(4)
        a.metric("Verdaderos positivos", int(tp)); b.metric("Falsos negativos", int(fn)); c.metric("Falsos positivos", int(fp)); d.metric("Verdaderos negativos", int(tn))
        st.write(f"**Interpretación:** {tp} abandonos fueron identificados correctamente; {fn} casos reales de abandono no fueron detectados; {fp} empleados fueron clasificados como abandono aunque permanecían. Los falsos negativos son relevantes si se busca identificar tantos abandonos como sea posible.")
        errs=[]
        for nm,(_,pr,_) in fitted.items():
            cmi=confusion_matrix(yte,pr); tni,fpi,fni,tpi=cmi.ravel()
            errs.append({"Modelo":nm,"Verdaderos positivos":tpi,"Falsos negativos":fni,"Falsos positivos":fpi,"Verdaderos negativos":tni})
        errdf=pd.DataFrame(errs)
        st.plotly_chart(px.bar(errdf, x="Modelo", y=["Verdaderos positivos","Falsos negativos","Falsos positivos"], barmode="group", title="Comparación de errores y aciertos entre modelos"), use_container_width=True)
    with tab2:
        figroc=go.Figure()
        figpr=go.Figure()
        for nm,(_,pr,prb) in fitted.items():
            fpr,tpr,_=roc_curve(yte,prb)
            prec,rec,_=precision_recall_curve(yte,prb)
            figroc.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"{nm} (AUC={roc_auc_score(yte,prb):.3f})"))
            figpr.add_trace(go.Scatter(x=rec,y=prec,mode="lines",name=f"{nm} (AP={average_precision_score(yte,prb):.3f})"))
        figroc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Referencia",line=dict(dash="dash")))
        figroc.update_layout(title="Curvas ROC: comparación de todos los modelos",xaxis_title="Tasa de falsos positivos",yaxis_title="Tasa de verdaderos positivos",xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1]))
        figpr.update_layout(title="Curvas Precision-Recall: comparación de todos los modelos",xaxis_title="Recall",yaxis_title="Precision",xaxis=dict(range=[0,1]),yaxis=dict(range=[0,1]))
        st.plotly_chart(figroc,use_container_width=True); st.plotly_chart(figpr,use_container_width=True)
        st.write("**Interpretación:** ROC-AUC resume la capacidad de ordenar casos positivos por encima de negativos. Precision-Recall se enfoca en el desempeño sobre la clase de abandono, minoritaria en esta base.")
    with tab3:
        estimator=pipe.named_steps["model"]
        names=pipe.named_steps["prep"].get_feature_names_out()
        if hasattr(estimator,"feature_importances_"):
            imp=pd.DataFrame({"Variable":names,"Importancia":estimator.feature_importances_}).sort_values("Importancia",ascending=False).head(20).sort_values("Importancia")
            st.plotly_chart(px.bar(imp,x="Importancia",y="Variable",orientation="h",title=f"20 variables más importantes: {chosen}"),use_container_width=True)
            st.write("**Interpretación:** estas variables contribuyen más a las decisiones internas del modelo de árbol. La importancia no implica causalidad.")
        elif hasattr(estimator,"coef_"):
            co=np.ravel(estimator.coef_)
            coef=pd.DataFrame({"Variable":names,"Coeficiente":co}).sort_values("Coeficiente",key=abs,ascending=False).head(20).sort_values("Coeficiente")
            st.plotly_chart(px.bar(coef,x="Coeficiente",y="Variable",orientation="h",title=f"Coeficientes principales: {chosen}"),use_container_width=True)
            st.write("**Interpretación:** el signo muestra la dirección de la asociación dentro del modelo y su magnitud depende del preprocesamiento. No representa una relación causal.")
        else:
            st.info("KNN no cuenta con importancia de variables nativa. Para interpretarlo se requiere aplicar una técnica adicional, como permutation importance.")
    with tab4:
        pca=PCA(n_components=2, random_state=42)
        # Use the preprocessed test matrix so PCA reflects the predictor space
        zmat=pipe.named_steps["prep"].transform(Xte)
        if hasattr(zmat,"toarray"): zmat=zmat.toarray()
        coords=pca.fit_transform(zmat)
        pcadf=pd.DataFrame({"Componente 1":coords[:,0],"Componente 2":coords[:,1],"Clase real":np.where(yte.to_numpy()==1,"Abandonó","Permanece"),"Predicción":np.where(pred==1,"Abandonó","Permanece")})
        st.plotly_chart(px.scatter(pcadf,x="Componente 1",y="Componente 2",color="Clase real",symbol="Predicción",title="PCA de los registros de prueba: clase real y predicción",opacity=.7),use_container_width=True)
        st.caption(f"Los dos componentes explican {pca.explained_variance_ratio_.sum():.1%} de la variabilidad del espacio transformado. Es una proyección para explorar patrones, no una métrica de desempeño.")

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
    st.subheader("Regresión del ingreso mensual")
    st.write("Se replica la selección del notebook: ingreso mensual como variable objetivo y edad, antigüedad, abandono, rol y departamento como predictores.")
    data = df[["Attrition","Age","MonthlyIncome","YearsAtCompany","JobRole","Department"]].copy()
    data = data.rename(columns={"Attrition":"Abandono","Age":"Edad","MonthlyIncome":"IngresoMensual","YearsAtCompany":"AñosEnCompañia","JobRole":"RolDeTrabajo","Department":"Departamento"})
    dm = pd.get_dummies(data, columns=["Abandono","RolDeTrabajo","Departamento"], drop_first=True)
    yreg = dm["IngresoMensual"]
    Xreg = dm.drop(columns=["IngresoMensual"])
    Xtr,Xte,ytr,yte = train_test_split(Xreg,yreg,test_size=.2,random_state=42)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
    regressors = {"Lineal": (LinearRegression(), Xtr, Xte), "Ridge": (Ridge(alpha=10), Xtr_s, Xte_s), "Lasso": (Lasso(alpha=10,max_iter=10000), Xtr_s, Xte_s)}
    rows=[]; fitted={}; predictions={}
    for name,(model,trainx,testx) in regressors.items():
        model.fit(trainx,ytr); pred=model.predict(testx); fitted[name]=(model,trainx,testx); predictions[name]=pred
        rows.append({"Modelo":name,"R²":r2_score(yte,pred),"RMSE":np.sqrt(mean_squared_error(yte,pred)),"MAE":mean_absolute_error(yte,pred),"MSE":mean_squared_error(yte,pred)})
    met=pd.DataFrame(rows)
    best=met.sort_values(["RMSE","MAE"],ascending=True).iloc[0]
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Mejor modelo por RMSE",best["Modelo"])
    c2.metric("R²",f"{best['R²']:.4f}")
    c3.metric("RMSE",f"${best['RMSE']:,.2f}")
    c4.metric("MAE",f"${best['MAE']:,.2f}")
    st.success(f"**Conclusión:** {best['Modelo']} obtiene el menor RMSE en esta ejecución. Su R² es {best['R²']:.3f}, por lo que explica cerca del {best['R²']:.1%} de la variabilidad del ingreso en el conjunto de prueba. La diferencia entre modelos es pequeña; se deben considerar también validación cruzada y residuos.")
    st.markdown("### Comparación de modelos")
    st.dataframe(met.style.format({"R²":"{:.6f}","RMSE":"{:,.2f}","MAE":"{:,.2f}","MSE":"{:,.0f}"}),use_container_width=True,hide_index=True)
    mlong=met.melt(id_vars="Modelo",value_vars=["R²"],var_name="Métrica",value_name="Valor")
    st.plotly_chart(px.bar(mlong,x="Modelo",y="Valor",text_auto=".4f",title="Comparación de R² (mayor es mejor)").update_yaxes(range=[0,1]),use_container_width=True)
    errlong=met.melt(id_vars="Modelo",value_vars=["RMSE","MAE"],var_name="Métrica",value_name="Error")
    st.plotly_chart(px.bar(errlong,x="Modelo",y="Error",color="Métrica",barmode="group",text_auto=".2f",title="Comparación de errores (menor es mejor)"),use_container_width=True)
    st.markdown("### Análisis gráfico por modelo")
    chosen=st.selectbox("Seleccione el modelo para ver sus gráficos e interpretación",["Ridge","Lineal","Lasso"],index=0,key="reg_model_select")
    pred=predictions[chosen]
    real_vs_pred=pd.DataFrame({"Ingreso real":yte.to_numpy(),"Ingreso estimado":pred})
    fig=px.scatter(real_vs_pred,x="Ingreso real",y="Ingreso estimado",opacity=.65,title=f"Ingreso real frente a estimado: {chosen}")
    lo=min(real_vs_pred.min()); hi=max(real_vs_pred.max())
    fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode="lines",name="Predicción ideal",line=dict(dash="dash")))
    st.plotly_chart(fig,use_container_width=True)
    resid=yte.to_numpy()-pred
    rdf=pd.DataFrame({"Ingreso estimado":pred,"Residuo":resid})
    st.plotly_chart(px.scatter(rdf,x="Ingreso estimado",y="Residuo",title=f"Diagnóstico de residuos: {chosen}").add_hline(y=0,line_dash="dash"),use_container_width=True)
    st.write(f"**Interpretación de {chosen}:** el MAE es ${met.loc[met['Modelo']==chosen,'MAE'].iloc[0]:,.2f}; en promedio, la diferencia absoluta entre el ingreso real y el estimado tiene ese orden de magnitud. Los residuos deben distribuirse alrededor de cero sin patrones sistemáticos; patrones visibles pueden señalar limitaciones del modelo.")
    model,_,_=fitted[chosen]
    coefs=np.ravel(model.coef_)
    coef=pd.DataFrame({"Variable":Xreg.columns,"Coeficiente":coefs}).sort_values("Coeficiente",key=abs,ascending=False).head(20).sort_values("Coeficiente")
    st.plotly_chart(px.bar(coef,x="Coeficiente",y="Variable",orientation="h",title=f"20 coeficientes de mayor magnitud: {chosen}"),use_container_width=True)
    st.write("**Lectura de coeficientes:** positivos se asocian con un ingreso estimado mayor y negativos con uno menor, manteniendo las demás variables constantes. Para Ridge y Lasso los predictores están estandarizados, por lo que las magnitudes no se interpretan directamente en las unidades originales. No representan causalidad.")

elif section == "Segmentación de empleados":
    st.subheader("Segmentación y perfiles de empleados")
    st.write("Se utilizan las cuatro variables del notebook: edad, ingreso mensual, años de experiencia y antigüedad. Se estandarizan antes de aplicar los métodos.")
    Xseg=df[["Age","MonthlyIncome","TotalWorkingYears","YearsAtCompany"]].copy()
    Xseg.columns=["Edad","Ingreso","Años_Experiencia","Antiguedad"]
    Z=StandardScaler().fit_transform(Xseg)
    # Reproducir configuraciones base del notebook para que la comparación sea interpretable
    labels_all={"K-means":KMeans(n_clusters=5,random_state=42,n_init=10).fit_predict(Z),
                "Jerárquico (Ward)":AgglomerativeClustering(n_clusters=5).fit_predict(Z),
                "DBSCAN":DBSCAN(eps=0.5,min_samples=5).fit_predict(Z)}
    scores={}
    for name,lab in labels_all.items():
        valid=lab!=-1 if name=="DBSCAN" else np.ones(len(lab),dtype=bool)
        uniq=set(lab[valid])
        # Para DBSCAN se replica el cálculo del notebook original, incluyendo la etiqueta -1.
        scores[name]=silhouette_score(Z,lab) if name=="DBSCAN" and len(set(lab))>1 else (silhouette_score(Z[valid],lab[valid]) if len(uniq)>1 and valid.sum()>len(uniq) else np.nan)
    score_df=pd.DataFrame([{"Método":n,"Silhouette":v,"Grupos":len(set(labels_all[n])-({-1} if n=="DBSCAN" else set())),"Ruido":int((labels_all[n]==-1).sum()) if n=="DBSCAN" else 0} for n,v in scores.items()])
    valid_scores=score_df.dropna(subset=["Silhouette"])
    best=valid_scores.sort_values("Silhouette",ascending=False).iloc[0] if not valid_scores.empty else None
    c1,c2,c3=st.columns(3)
    if best is not None:
        c1.metric("Mejor Silhouette",best["Método"],f"{best['Silhouette']:.4f}")
        c2.metric("Grupos del método mejor puntuado",int(best["Grupos"]))
    else:
        c1.metric("Mejor método","No calculable")
        c2.metric("Grupos","—")
    c3.metric("Ruido DBSCAN",f"{int((labels_all['DBSCAN']==-1).sum())} ({(labels_all['DBSCAN']==-1).mean():.2%})")
    if best is not None:
        st.success(f"**Método con mayor Silhouette: {best['Método']} ({best['Silhouette']:.4f}).** Es el mejor según este criterio entre las tres configuraciones comparadas. El valor indica la cohesión y separación relativas; no prueba que los grupos sean útiles para todos los propósitos.")
    st.markdown("### Comparación de métodos")
    st.dataframe(score_df.style.format({"Silhouette":"{:.4f}"}),use_container_width=True,hide_index=True)
    st.plotly_chart(px.bar(score_df,x="Método",y="Silhouette",text_auto=".4f",title="Comparación de Silhouette: mayor es mejor"),use_container_width=True)
    st.markdown("### Explorar cada método")
    algo=st.selectbox("Seleccione el método que desea analizar",["K-means","Jerárquico (Ward)","DBSCAN"],index=0,key="cluster_method_select")
    labels=labels_all[algo]
    n_clusters=len(set(labels)-({-1} if algo=="DBSCAN" else set()))
    noise=int((labels==-1).sum()) if algo=="DBSCAN" else 0
    out=Xseg.copy(); out["Grupo"]=labels.astype(str)
    profile=out[out["Grupo"]!="-1"].groupby("Grupo")[Xseg.columns].agg(["count","mean"]).round(2)
    st.markdown(f"#### Perfil de grupos: {algo}")
    st.dataframe(profile,use_container_width=True)
    a,b,c=st.columns(3)
    a.metric("Grupos identificados",n_clusters)
    a_score=scores[algo]
    b.metric("Silhouette",f"{a_score:.4f}" if np.isfinite(a_score) else "No calculable")
    c.metric("Observaciones de ruido",f"{noise:,}" if algo=="DBSCAN" else "No aplica")
    if np.isfinite(a_score):
        st.write("**Interpretación:** " + ("la separación de los grupos es relativamente débil/moderada y existe solapamiento entre algunos perfiles." if a_score<.5 else "los grupos presentan una separación relativamente marcada según Silhouette."))
    if algo=="DBSCAN":
        st.write(f"DBSCAN identificó {n_clusters} grupos y {noise} observaciones como ruido ({noise/len(df):.2%}). El ruido corresponde a registros que no cumplen el criterio de densidad configurado; no significa que sean errores.")
    else:
        st.write("Las etiquetas numéricas de los grupos son arbitrarias. Para describir cada perfil, compare sus promedios de edad, ingreso, experiencia y antigüedad; no debe inferirse nivel de desempeño ni riesgo sin evidencia adicional.")
    pca=PCA(n_components=2,random_state=42); coords=pca.fit_transform(Z)
    plot=pd.DataFrame({"Componente 1":coords[:,0],"Componente 2":coords[:,1],"Grupo":labels.astype(str)})
    st.plotly_chart(px.scatter(plot,x="Componente 1",y="Componente 2",color="Grupo",title=f"Proyección PCA de empleados: {algo}",opacity=.7),use_container_width=True)
    st.caption(f"Los dos componentes PCA explican conjuntamente {pca.explained_variance_ratio_.sum():.1%} de la varianza estandarizada. La proyección puede ocultar estructura presente en las dimensiones originales.")
    # Side-by-side comparison of the three methods, similar to the notebook
    st.markdown("### Comparación visual de los tres métodos")
    fig=go.Figure()
    for method,lab in labels_all.items():
        fig.add_trace(go.Scatter(x=coords[:,0],y=coords[:,1],mode="markers",name=method,visible=(method==algo),marker=dict(size=6),text=lab.astype(str),hovertemplate="PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>Grupo=%{text}<extra></extra>"))
    fig.update_layout(title="PCA: seleccione la leyenda para mostrar/ocultar métodos",xaxis_title="Componente 1",yaxis_title="Componente 2")
    st.plotly_chart(fig,use_container_width=True)
    st.download_button("Descargar empleados con grupo asignado",out.to_csv(index=False).encode("utf-8-sig"),f"empleados_segmentados_{algo.replace(' ','_')}.csv","text/csv")

else:
    st.subheader("Metodología y limitaciones")
    etapas = [
        ("1. Datos", "Se utiliza la base IBM HR Analytics, con 1.470 registros y variables demográficas, laborales y salariales."),
        ("2. Preprocesamiento", "Se revisan tipos de variables y se transforman las variables categóricas mediante codificación. En clasificación se aplica preprocesamiento dentro del pipeline; en regresión se emplea la codificación dummy definida en el notebook; en segmentación se estandarizan las cuatro variables numéricas seleccionadas."),
        ("3. Modelamiento", "Clasificación: Regresión Logística, KNN, Árbol de Decisión y Random Forest. Regresión: Lineal, Ridge y Lasso. Segmentación: K-means, jerárquico y DBSCAN."),
        ("4. Optimización y selección", "Los modelos se comparan con métricas apropiadas a cada tarea. Para regresión se consideran R², MAE y RMSE; para segmentación se compara Silhouette; para clasificación se revisan Recall, F1, ROC-AUC y PR-AUC, además de la matriz de confusión."),
        ("5. Evaluación e interpretación", "Cada módulo presenta tablas, visualizaciones y una interpretación automática que identifica el mejor resultado según el criterio definido. La selección se refiere a los modelos y parámetros ejecutados en la aplicación."),
        ("6. Limitaciones", "Los resultados dependen de las variables, particiones y parámetros empleados. Las asociaciones no demuestran causalidad, las etiquetas de grupo son arbitrarias y las predicciones no deben utilizarse como único criterio para decisiones laborales.")
    ]
    for titulo, descripcion in etapas:
        with st.container(border=True):
            st.subheader(titulo)
            st.write(descripcion)
    st.info("Referencia a los resultados de los notebooks: Ridge obtuvo el menor RMSE en la evaluación de regresión registrada (1.943,80 aproximadamente); K-means obtuvo Silhouette de 0,3363, superior al jerárquico (0,3079) y DBSCAN (0,3007). En clasificación, la aplicación recalcula métricas en su partición de prueba y pueden diferir de las cifras del notebook original.")
    st.write("Base usada:", f"{len(df):,} filas y {df.shape[1]} columnas.")
    st.write("Valores faltantes totales:", int(df.isna().sum().sum()))
    st.write("Registros duplicados:", int(df.duplicated().sum()))
