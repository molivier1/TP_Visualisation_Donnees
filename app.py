import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from Clement import *

st.set_page_config(page_title="Transactions Dashboard", layout="wide")

@st.cache_data
def load_data():
    clients_a_contacter = pd.read_csv("Data/clients_a_contacter.csv")
    train_info = pd.read_csv("Data/train_info.csv")

    return (clients_a_contacter, train_info)

clients_a_contacter, train_info = load_data()

st.title("TP - Visualisation et exploration de données")

tabs = st.tabs(["Résumé", "Visualisation", "Analyse"])

with tabs[0]:
    st.header("Résumé du jeu de données et typologie des variables")
    #st.info("À compléter")
    st.subheader("Les 10 premières lignes du Dataframe train_info :")
    st.dataframe(train_info.head(10))

    lignes, colonnes = clients_shape(train_info)
    st.write(f"Dimensions du Dataframe 'train_info' -> {lignes} lignes et {colonnes} colonnes")

    st.subheader("Types des variables :")
    st.write(clients_type(train_info))

with tabs[1]:
    st.subheader("Visualisation des variables catégorielles")
    st.info("À compléter")

    st.subheader("Visualisation des variables quantitatives")
    st.info("À compléter")

with tabs[2]:
    st.subheader("Analyse de corrélation")
    st.info("À compléter")

    st.subheader("Analyse des relations entre variables")
    st.info("À compléter")

    st.subheader("Analyse de la variable cible")
    st.info("À compléter")
