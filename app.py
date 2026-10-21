    else:
        columnas_seg = [variables_disponibles[v] for v in variables_elegidas]
        Xseg = df[columnas_seg].copy()
        Xseg.columns = variables_elegidas
        Z = StandardScaler().fit_transform(Xseg)

        labels_all = {
            "K-means": KMeans(n_clusters=k_grupos, random_state=42, n_init=10).fit_predict(Z),
            "Jerárquico (Ward)": AgglomerativeClustering(n_clusters=k_grupos).fit_predict(Z),
            "DBSCAN": DBSCAN(eps=eps if metodo_explorar == "DBSCAN" else 0.5,
                             min_samples=min_samples if metodo_explorar == "DBSCAN" else 5).fit_predict(Z),
        }
        scores = {}
        for name, lab in labels_all.items():
            # En DBSCAN, Silhouette se calcula sobre los grupos válidos, excluyendo ruido.
            valid = lab != -1 if name == "DBSCAN" else np.ones(len(lab), dtype=bool)
            etiquetas_validas = lab[valid]
            grupos_validos = np.unique(etiquetas_validas)
            if len(grupos_validos) >= 2 and valid.sum() > len(grupos_validos):
                scores[name] = silhouette_score(Z[valid], etiquetas_validas)
            else:
                scores[name] = np.nan

        score_df = pd.DataFrame([
            {
                "Método": nombre,
                "Silhouette": scores[nombre],
                "Grupos": len(set(labels_all[nombre]) - ({-1} if nombre == "DBSCAN" else set())),
                "Ruido": int((labels_all[nombre] == -1).sum()) if nombre == "DBSCAN" else 0,
            }
            for nombre in labels_all
        ])
        valid_scores = score_df.dropna(subset=["Silhouette"])
        best = valid_scores.sort_values("Silhouette", ascending=False).iloc[0] if not valid_scores.empty else None

        c1, c2, c3 = st.columns(3)
        if best is not None:
            c1.metric("Mejor método por Silhouette", best["Método"], f"{best['Silhouette']:.4f}")
            c2.metric("Grupos del mejor método", int(best["Grupos"]))
        else:
            c1.metric("Mejor método", "No calculable")
            c2.metric("Grupos", "—")
        ruido_dbscan = int((labels_all["DBSCAN"] == -1).sum())
        c3.metric("Ruido DBSCAN", f"{ruido_dbscan} ({(labels_all['DBSCAN'] == -1).mean():.2%})")
        if best is not None:
            st.success(
                f"**Mejor método: {best['Método']}.** Obtuvo el mayor coeficiente Silhouette válido "
                f"({best['Silhouette']:.4f}) entre los tres métodos evaluados e identificó "
                f"{int(best['Grupos'])} grupos. La comparación utiliza las variables y los parámetros seleccionados."
            )
        else:
            st.warning("No fue posible calcular un coeficiente Silhouette válido para los métodos evaluados con esta configuración.")

        st.markdown("### Comparación de métodos")
        st.dataframe(score_df.style.format({"Silhouette": "{:.4f}"}), use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(score_df, x="Método", y="Silhouette", text_auto=".4f", title="Comparación de Silhouette: mayor es mejor"),
            use_container_width=True,
        )
        st.markdown("### Explorar cada método")
        algo = metodo_explorar
        labels = labels_all[algo]
        n_clusters = len(set(labels) - ({-1} if algo == "DBSCAN" else set()))
        noise = int((labels == -1).sum()) if algo == "DBSCAN" else 0
        out = Xseg.copy()
        out["Grupo"] = labels.astype(str)
        profile = out[out["Grupo"] != "-1"].groupby("Grupo")[Xseg.columns].agg(["count", "mean"]).round(2)
        st.markdown(f"#### Perfil de grupos: {algo}")
        st.dataframe(profile, use_container_width=True)
        a, b, c = st.columns(3)
        a.metric("Grupos identificados", n_clusters)
        a_score = scores[algo]
        b.metric("Silhouette", f"{a_score:.4f}" if np.isfinite(a_score) else "No calculable")
        c.metric("Observaciones de ruido", f"{noise:,}" if algo == "DBSCAN" else "No aplica")
        if np.isfinite(a_score):
            st.write("**Interpretación:** " + ("la separación de los grupos es relativamente débil/moderada y existe solapamiento entre algunos perfiles." if a_score < .5 else "los grupos presentan una separación relativamente marcada según Silhouette."))
        if algo == "DBSCAN":
            st.write(f"DBSCAN identificó {n_clusters} grupos y {noise} observaciones como ruido ({noise/len(df):.2%}). El ruido corresponde a registros que no cumplen el criterio de densidad configurado; no significa que sean errores.")
        else:
            st.write("Las etiquetas numéricas de los grupos son arbitrarias. Compare los promedios de las variables seleccionadas para describir los perfiles; no debe inferirse desempeño ni riesgo sin evidencia adicional.")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(Z)
        plot = pd.DataFrame({"Componente 1": coords[:, 0], "Componente 2": coords[:, 1], "Grupo": labels.astype(str)})
        st.plotly_chart(px.scatter(plot, x="Componente 1", y="Componente 2", color="Grupo", title=f"Proyección PCA de empleados: {algo}", opacity=.7), use_container_width=True)
        st.caption(f"Los dos componentes PCA explican conjuntamente {pca.explained_variance_ratio_.sum():.1%} de la varianza estandarizada. La proyección puede ocultar estructura presente en las dimensiones originales.")
        st.markdown("### Comparación visual de los tres métodos")
        fig = go.Figure()
        for method, lab in labels_all.items():
            fig.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers", name=method, visible=(method == algo), marker=dict(size=6), text=lab.astype(str), hovertemplate="PC1=%{x:.2f}<br>PC2=%{y:.2f}<br>Grupo=%{text}<extra></extra>"))
        fig.update_layout(title="PCA: seleccione la leyenda para mostrar/ocultar métodos", xaxis_title="Componente 1", yaxis_title="Componente 2")
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Descargar empleados con grupo asignado", out.to_csv(index=False).encode("utf-8-sig"), f"empleados_segmentados_{algo.replace(' ', '_')}.csv", "text/csv")

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
