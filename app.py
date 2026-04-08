import pandas as pd
import streamlit as st

from Clement import *

st.set_page_config(page_title="Transactions Dashboard", layout="wide")

@st.cache_data
def load_data():
    clients_a_contacter = pd.read_csv("Data/clients_a_contacter.csv")
    train_info = pd.read_csv("Data/train_info.csv")

    return (clients_a_contacter, train_info)

clients_a_contacter, train_info = load_data()

variables_categorielles = [
    "genre",
    "permis_conduire",
    "ancien_assure",
    "age_vehicule",
    "vehicule_endommage",
]

st.title("TP - Visualisation et exploration de données")

tabs = st.tabs(["Résumé", "Visualisation", "Analyse"])

with tabs[0]:
    st.header("Résumé du jeu de données et typologie des variables")

    st.subheader("Les 10 premières lignes du Dataframe train_info :")
    st.dataframe(train_info.head(10))

    lignes, colonnes = clients_shape(train_info)
    st.write(f"Dimensions du Dataframe 'train_info' -> {lignes} lignes et {colonnes} colonnes")

    st.subheader("Types des variables :")
    st.write(clients_type(train_info))

    st.write(f"Valeurs manquantes : {clients_valeur_manquantes(train_info).size}")

    st.write(f"\nValeurs duppliquées : {clients_duplicated_values(train_info).size}")

with tabs[1]:
    st.subheader("Visualisation des variables catégorielles")

    st.write("Variables catégorielles principales :")
    st.write(variables_categorielles)

    st.write("Analyse de la distribution des principales variables catégorielles et leur relation avec la variable cible reponse_client")

    for variable in variables_categorielles:
        st.write(f"Count plot de {variable} selon reponse_client :")
        st.pyplot(clients_countplot(train_info, variable))

        st.write(f"Pourcentage de réponses positives pour {variable} :")
        st.write(clients_reponse_par_modalite(train_info, variable))

    st.subheader("Visualisation des variables quantitatives")
    st.info("À compléter")

with tabs[2]:
    st.subheader("Analyse de corrélation")
    st.info("À compléter")

    st.subheader("Analyse des relations entre variables")
    st.info("À compléter")

    st.subheader("Analyse de la variable cible")
    st.info("À compléter")
