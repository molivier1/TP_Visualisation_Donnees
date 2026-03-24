import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Transactions Dashboard", layout="wide")

@st.cache_data
def load_data():
    clients_a_contacter = pd.read_csv("Data/clients_a_contacter.csv")
    train_info = pd.read_csv("Data/train_info.csv")

    return (clients_a_contacter, train_info)

clients_a_contacter, train_info = load_data()

with st.sidebar:
    """st.title("Filtres")

    store_options = sorted(df["store_type"].dropna().unique().tolist())
    store_sel = st.multiselect("Type de magasin", store_options, default=store_options)

    dmin, dmax = df["tran_date"].min(), df["tran_date"].max()
    start, end = st.date_input("Période", value=(dmin.date(), dmax.date()))

    include_returns = st.toggle("Inclure les retours", value=True)

    # Exercice 1
    st.subheader("Exo 1 : Distributions")

    var_num = st.selectbox("Variable numérique", ["qty", "rate", "tax", "total_amt"])
    # Trouver combien mettre en val par défaut??
    nb_classes = st.slider("Nombre de classes", value=50)"""

#df_f = df[df["store_type"].isin(store_sel)].copy()
#df_f = df_f[(df_f["tran_date"] >= pd.to_datetime(start)) & (df_f["tran_date"] <= pd.to_datetime(end))]

"""if not include_returns:
    df_f = df_f[df_f["total_amt"] >= 0]"""

st.title("📊 Dashboard Transactions (dataset fusionné)")

#c1, c2, c3, c4 = st.columns(4)
#c1.metric("Transactions", int(df_f.shape[0]))
#c2.metric("Clients uniques", int(df_f["cust_id"].nunique()))
#c3.metric("Ventes nettes", float(df_f["total_amt"].sum()))
#c4.metric("Retours (nb)", int((df_f["total_amt"] < 0).sum()))

tabs = st.tabs(["Aperçu", "Distributions", "Temps", "Catégories", "Clients", "Qualité"])

with tabs[0]:
    st.subheader("Aperçu des données filtrées")
    """st.dataframe(df_f.head(100))

    st.subheader("Téléchargement")
    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger les données filtrées",
        data=csv,
        file_name="transactions_filtrees.csv",
        mime="text/csv"
    )"""

# Exercice 1
with tabs[1]:
    st.subheader("Distributions")
    #st.info("À compléter")
    """
    # Histogramme interactif sur une variable numérique 
    # choisie parmi : qty, rate, tax et total_amt
    st.subheader("Histogramme")
    # Je fais l'histogramme en fonction des valeurs dans la sidebar...
    histogramme = px.histogram(df_f, x = var_num, nbins = nb_classes, 
                               title = f"Distribution de {var_num}")
    # Et . . . son affichage !
    st.plotly_chart(histogramme)


    # Un boxplot par store_type
    st.subheader("Boxplot")
    # Les store_type sont déjà filtrés donc juste à préciser
    # la variable numérique déjà obtenu
    boxplot = px.box(df_f, x = "store_type", y = var_num,
                     title=f"{var_num} en fonction du store_type")
    st.plotly_chart(boxplot)"""

# Exercice 2
with tabs[2]:
    st.subheader("Temps")
    #st.info("À compléter")
    # Ici je compte le nb de transactions et le montant total par mois
    # reset_index() permet juste de faire un vrai index et pas la date
    """
    df_temps = df_f.groupby(pd.Grouper(key = "tran_date", freq = "MS")).agg({
        "cust_id": "count",
        "total_amt": "sum"
    }).reset_index()

    # Je renomme la colonne psk cust_id est pas ultra accurate
    # le vrai objectif était le nb de transactions
    df_temps = df_temps.rename(columns={"cust_id": "nb_transactions"})
    #st.dataframe(df_temps.head(100))

    # Une courbe Plotly du montant total mensuel (total_amt)
    st.subheader("Montant total mensuel")
    # Ici j'prends la date de transaction et le montant total
    # pour simplement faire la courbe
    courbe_total_mensuel = px.line(df_temps, x = "tran_date", 
                                   y = "total_amt",
                                   title = "Courbe montant total mensuel")
    
    st.plotly_chart(courbe_total_mensuel)
    
    # Une seconde courbe du nombre de transactions
    # Même opérations mais cette fois je travail sur nb_transactions
    courbe_nb_transactions = px.line(df_temps, x = "tran_date", 
                                   y = "nb_transactions",
                                   title = "Courbe nb transactions mensuel")
    
    st.plotly_chart(courbe_nb_transactions)
    """


"""
with tabs[3]:
    st.subheader("Catégories")
    st.info("À compléter")

with tabs[4]:
    st.subheader("Clients")
    st.info("À compléter")

with tabs[5]:
    st.subheader("Qualité des données")
    st.info("À compléter")
"""
