import pandas as pd
import streamlit as st
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from fonctions import *

st.set_page_config(page_title="Transactions Dashboard", layout="wide")


@st.cache_data
def load_data():
    clients_a_contacter = pd.read_csv("Data/clients_a_contacter.csv")
    train_info = pd.read_csv("Data/train_info.csv")
    return clients_a_contacter, train_info


def afficher_plot(plot, width="stretch"):
    st.plotly_chart(plot, width=width)


clients_a_contacter, train_info = load_data()

lignes, colonnes = clients_shape(train_info)
missing_count = train_info.isnull().sum().sum()
dup_count = clients_duplicated_rows_count(train_info)

variables_categorielles = [
    "genre",
    "permis_conduire",
    "ancien_assure",
    "age_vehicule",
    "vehicule_endommage",
]

variables_quantitatives = [
    "age",
    "prime_annuelle",
    "anciennete",
]

st.title("TP - Visualisation et exploration de données")

with st.sidebar:
    st.header("Aperçu du dataset")
    st.metric("Nombre de lignes", f"{lignes:,}")
    st.metric("Nombre de colonnes", colonnes)
    st.metric("Valeurs manquantes", missing_count)
    st.metric("Doublons", dup_count)

    st.divider()
    st.subheader("Types de données")
    types_df = (
        clients_type(train_info)
        .astype(str)
        .rename("dtype")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    st.dataframe(types_df, width="stretch")


tabs = st.tabs(["Résumé", "Visualisation", "Analyse", "Modélisation", "Stratégie marketing"])

with tabs[0]:
    st.subheader("Aperçu des données")
    st.dataframe(train_info.head(50), width="stretch")

    col_missing, col_outliers = st.columns(2)

    with col_missing:
        st.subheader("Valeurs manquantes")
        st.dataframe(
            clients_missing_values_summary(train_info).reset_index(),
            width="stretch",
        )

    with col_outliers:
        st.subheader("Valeurs aberrantes")
        outliers_df = clients_outliers_summary(train_info, variables_quantitatives)
        st.dataframe(outliers_df, width="stretch")

    st.caption(
        "On remarque surtout quelques valeurs élevées pour la prime annuelle. "
        "On les garde car elles peuvent correspondre à de vrais profils clients."
    )

with tabs[1]:
    st.header("Analyse univariée")

    mode_analyse = st.radio(
        "Type de variable à explorer :",
        ["Catégorielles", "Quantitatives"],
        horizontal=True,
    )

    if mode_analyse == "Catégorielles":
        var_cat = st.selectbox(
            "Choisir une variable catégorielle :",
            variables_categorielles,
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(f"**Distribution de {var_cat}**")
            afficher_plot(clients_countplot(train_info, var_cat))
        with c2:
            st.write("**Taux de réponse par modalité (%)**")
            st.dataframe(
                clients_reponse_par_modalite(train_info, var_cat).reset_index(),
                width="stretch",
            )

    else:
        var_quant = st.selectbox(
            "Choisir une variable quantitative :",
            variables_quantitatives,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Histogramme : {var_quant}**")
            afficher_plot(clients_histplot(train_info, var_quant))
        with c2:
            st.write(f"**Boxplot : {var_quant}**")
            afficher_plot(clients_boxplot(train_info, var_quant))

        st.write(f"**Courbe de densité : {var_quant}**")
        afficher_plot(clients_kdeplot(train_info, var_quant))

with tabs[2]:
    col_target, col_corr = st.columns([1, 1.5])

    with col_target:
        st.subheader("Distribution cible")
        afficher_plot(clients_target_distribution(train_info))
        nb_negatifs = int((train_info["reponse_client"] == 0).sum())
        nb_positifs = int((train_info["reponse_client"] == 1).sum())
        taux_positifs = train_info["reponse_client"].mean() * 100
        st.caption(
            f"Classes observées : {nb_negatifs:,} non intéressés et "
            f"{nb_positifs:,} intéressés ({taux_positifs:.2f}% de classe positive)."
        )

    with col_corr:
        st.subheader("Matrice de corrélation")
        afficher_plot(clients_correlation_matrix(train_info))

    st.divider()
    st.subheader("Analyse des relations entre variables")
    cx, cy = st.columns(2)
    var_x = cx.selectbox("Variable X", variables_quantitatives, index=0, key="x_scatter")
    var_y = cy.selectbox("Variable Y", variables_quantitatives, index=1, key="y_scatter")
    afficher_plot(clients_scatter_relation(train_info, var_x, var_y))

with tabs[3]:
    st.header("Modélisation et prédictions")

    if st.button("Lancer l'entraînement du modèle"):
        with st.spinner("Transformation des données et entraînement..."):
            df_model_prep, preprocessing_artifacts = preparer_dataset_complet(
                train_info, is_train=True
            )

            st.session_state["preprocessing_artifacts"] = preprocessing_artifacts

            model, scaler, X_test, y_test, feature_names, best_params, best_cv_f1 = (
                entrainer_modele_rf(df_model_prep)
            )

            st.session_state["model"] = model
            st.session_state["scaler"] = scaler
            st.session_state["feature_names"] = feature_names

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            pr_auc = average_precision_score(y_test, y_proba)

            st.subheader("Lecture du déséquilibre de classes")
            st.info(
                "La classe positive est minoritaire, donc on regarde surtout "
                "les métriques comme F1, ROC-AUC et PR-AUC."
            )

            m1, m2, m3 = st.columns(3)
            m1.metric("F1-score", f"{f1:.3f}")
            m2.metric("ROC-AUC", f"{roc_auc:.3f}")
            m3.metric("PR-AUC", f"{pr_auc:.3f}")

            st.subheader("Petit réglage des hyperparamètres")
            st.write(
                {
                    "meilleurs_parametres": best_params,
                    "meilleur_f1_cv": round(best_cv_f1, 4),
                }
            )

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Performances du modèle")
                st.code(classification_report(y_test, y_pred))

            with c2:
                st.subheader("Importance des variables")
                importances = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Importance": model.feature_importances_,
                    }
                ).sort_values(by="Importance", ascending=True)

                fig_imp = px.bar(
                    importances,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    color_discrete_sequence=["#F58518"],
                )
                st.plotly_chart(fig_imp, width="stretch")

            c3, c4 = st.columns(2)
            with c3:
                afficher_plot(clients_confusion_matrix_figure(y_test, y_pred))
                afficher_plot(clients_roc_curve_figure(y_test, y_proba))

            with c4:
                afficher_plot(clients_pr_curve_figure(y_test, y_proba))

with tabs[4]:
    st.header("Optimisation du ciblage marketing")
    st.write(
        "L'idée ici est de repérer les clients qui ont une probabilité intermédiaire, "
        "car ce sont souvent ceux sur lesquels une action commerciale peut faire la différence."
    )

    if "model" not in st.session_state:
        st.warning("Veuillez d'abord lancer l'entraînement dans l'onglet 'Modélisation'.")
    else:
        if st.button("Générer la liste de contact"):
            model = st.session_state["model"]
            scaler = st.session_state["scaler"]
            feature_names = st.session_state["feature_names"]
            preprocessing_artifacts = st.session_state["preprocessing_artifacts"]

            liste_finale = generer_predictions_marketing(
                model,
                scaler,
                clients_a_contacter,
                feature_names,
                preprocessing_artifacts,
            )

            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Répartition")
                fig_pie = px.pie(
                    liste_finale,
                    names="strategie_marketing",
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                st.plotly_chart(fig_pie, width="stretch")

            with col2:
                st.subheader("Profil des cibles")
                cibles = liste_finale[
                    liste_finale["strategie_marketing"].str.contains("CIBLER", na=False)
                ]
                st.metric("Clients prioritaires", len(cibles))
                st.write(f"Âge moyen : {round(cibles['age'].mean(), 1)} ans")
                st.write(
                    f"Prime annuelle moyenne : {round(cibles['prime_annuelle'].mean(), 1)}"
                )

            st.divider()
            st.subheader("Profil synthétique des clients à cibler")
            st.dataframe(clients_marketing_profile_summary(cibles), width="stretch")

            st.subheader("Répartition des cibles par profil")
            p1, p2, p3 = st.columns(3)
            with p1:
                afficher_plot(
                    px.histogram(cibles, x="genre", color="genre", title="Genre des cibles")
                )
            with p2:
                afficher_plot(
                    px.histogram(
                        cibles,
                        x="age_vehicule",
                        color="age_vehicule",
                        title="Âge du véhicule",
                    )
                )
            with p3:
                afficher_plot(
                    px.histogram(
                        cibles,
                        x="vehicule_endommage",
                        color="vehicule_endommage",
                        title="Véhicule endommagé",
                    )
                )

            st.divider()
            st.subheader("Liste prioritaire")
            st.dataframe(
                cibles.sort_values(by="probabilite_souscription", ascending=False),
                width="stretch",
            )

            csv = cibles.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger CSV",
                data=csv,
                file_name="marketing_priorite.csv",
            )
