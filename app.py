import pandas as pd
import streamlit as st

from fonctions import *

st.set_page_config(page_title="Transactions Dashboard", layout="wide")


@st.cache_data
def load_data():
    clients_a_contacter = pd.read_csv("Data/clients_a_contacter.csv")
    train_info = pd.read_csv("Data/train_info.csv")

    return (clients_a_contacter, train_info)


def afficher_plot(plot, use_container_width=True):
    """Affiche un plot Plotly dans Streamlit."""
    st.plotly_chart(plot, use_container_width=use_container_width)

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
    st.header("Aperçu du Dataset")
    st.metric("Nombre de lignes", f"{lignes:,}")
    st.metric("Nombre de colonnes", colonnes)
    st.metric("Valeurs manquantes", missing_count)
    st.metric("Doublons", dup_count)
    
    st.divider()
    st.subheader("Types de données")
    st.write(clients_type(train_info))

tabs = st.tabs(["Résumé", "Visualisation", "Analyse", "Modélisation", "Stratégie Marketing"])

with tabs[0]:
    st.subheader("Aperçu des données")
    st.dataframe(train_info.head(50), use_container_width=True)
    
    col_missing, col_types = st.columns(2)
    with col_missing:
        st.subheader("Détail valeurs manquantes")
        st.dataframe(clients_missing_values_summary(train_info))

with tabs[1]:
    st.header("Analyse Univariée")
    
    mode_analyse = st.radio("Type de variable à explorer :", ["Catégorielles", "Quantitatives"], horizontal=True)
    
    if mode_analyse == "Catégorielles":
        var_cat = st.selectbox("Choisir une variable catégorielle :", variables_categorielles)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.write(f"**Distribution de {var_cat}**")
            afficher_plot(clients_countplot(train_info, var_cat))
        with c2:
            st.write("**Taux de réponse par modalité (%)**")
            st.dataframe(clients_reponse_par_modalite(train_info, var_cat), use_container_width=True)
            
    else:
        var_quant = st.selectbox("Choisir une variable quantitative :", variables_quantitatives)
        
        # Mise en page en grille pour les variables numériques
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.write(f"**Histogramme : {var_quant}**")
            afficher_plot(clients_histplot(train_info, var_quant))
        with row1_col2:
            st.write(f"**Boxplot : {var_quant}**")
            afficher_plot(clients_boxplot(train_info, var_quant))
            
        st.write(f"**Courbe de densité (KDE) : {var_quant}**")
        afficher_plot(clients_kdeplot(train_info, var_quant))

    

with tabs[2]:
    col_target, col_corr = st.columns([1, 1.5])
    
    with col_target:
        st.subheader("Distribution Cible")
        afficher_plot(clients_target_distribution(train_info))

    with col_corr:
        st.subheader("Matrice de Corrélation")
        afficher_plot(clients_correlation_matrix(train_info))
    
    st.divider()
    st.subheader("Analyse des relations entre variables")
    cx, cy = st.columns(2)
    var_x = cx.selectbox("Variable X", variables_quantitatives, index=0, key="x_scatter")
    var_y = cy.selectbox("Variable Y", variables_quantitatives, index=1, key="y_scatter")
    
    afficher_plot(clients_scatter_relation(train_info, var_x, var_y))

with tabs[3]:
    st.header("Intelligence Artificielle & Prédictions")
    
    if st.button("Lancer l'entraînement du modèle"):
        with st.spinner("Transformation des données et entraînement..."):
            # CORRECTION 1 : Bien récupérer le tuple (DF, Mappings)
            df_model_prep, mappings_entrainement = preparer_dataset_complet(train_info, is_train=True)
            
            # Sauvegarde dans la session pour l'onglet suivant
            st.session_state['mappings'] = mappings_entrainement
            
            # CORRECTION 2 : Bien récupérer les 5 éléments du modèle
            model, scaler, X_test, y_test, feature_names = entrainer_modele_rf(df_model_prep)
            
            # Sauvegarde des objets du modèle
            st.session_state['model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['feature_names'] = feature_names

            # 3. Métriques
            from sklearn.metrics import classification_report
            y_pred = model.predict(X_test)
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Performances du modèle")
                st.code(classification_report(y_test, y_pred))
            
            with c2:
                st.subheader("Importance des variables")
                importances = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=True)
                
                fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h',
                                 color_discrete_sequence=["#F58518"])
                st.plotly_chart(fig_imp, use_container_width=True)


with tabs[4]:
    st.header("Optimisation du Ciblage Marketing")
    
    # On vérifie si le modèle a été entraîné au préalable
    if 'model' not in st.session_state:
        st.warning("Veuillez d'abord lancer l'entraînement dans l'onglet 'Modélisation'.")
    else:
        if st.button("Générer la liste de contact"):
            # CORRECTION 3 : On utilise les objets stockés dans la session
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            feature_names = st.session_state['feature_names']
            mappings = st.session_state['mappings']
            
            # Appel de la fonction de prédiction
            liste_finale = generer_predictions_marketing(
                model, 
                scaler, 
                clients_a_contacter, 
                feature_names, 
                mappings
            )
            
            # --- Affichage des résultats ---
            col1, col2 = st.columns([1, 2])
            with col1:
                st.subheader("Répartition")
                fig_pie = px.pie(liste_finale, names='strategie_marketing', color_discrete_sequence=px.colors.qualitative.Safe)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Profil des cibles")
                cibles = liste_finale[liste_finale['strategie_marketing'] == "À CIBLER (Zone d'influence)"]
                st.metric("Clients prioritaires", len(cibles))
                st.write(f"Âge moyen : {round(cibles['age'].mean(), 1)} ans")
            
            st.divider()
            st.subheader("Liste prioritaires")
            st.dataframe(cibles.sort_values(by='probabilite_souscription', ascending=False))
            
            csv = cibles.to_csv(index=False).encode('utf-8')
            st.download_button("Télécharger CSV", data=csv, file_name="marketing_priorite.csv")