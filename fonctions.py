import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def clients_shape(clients):
    return clients.shape

def clients_type(clients):
    return clients.dtypes

def clients_missing_values_summary(clients):
    return clients.isnull().sum()

def clients_duplicated_rows_count(clients):
    return clients.duplicated().sum()

def clients_countplot(clients, variable, cible="reponse_client"):
    dataframe = clients[[variable, cible]].copy()
    dataframe[variable] = dataframe[variable].fillna("Manquant")
    dataframe[cible] = dataframe[cible].map({0: "Negative", 1: "Positive"})
    
    fig = px.histogram(
        dataframe, 
        x=variable, 
        color=cible, 
        barmode='group',
        title=f"Distribution de {variable} selon {cible}",
        color_discrete_map={"Negative": "#4C78A8", "Positive": "#F58518"},
        category_orders={variable: dataframe[variable].value_counts().index.tolist()}
    )
    return fig


def clients_reponse_par_modalite(clients, variable, cible="reponse_client"):
    return (
        clients.groupby(variable)[cible]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .round(2)
    )


def clients_histplot(clients, variable, cible="reponse_client"):
    dataframe = clients[[variable, cible]].copy()
    dataframe[cible] = dataframe[cible].map({0: "Negative", 1: "Positive"})

    fig = px.histogram(
        dataframe,
        x=variable,
        color=cible,
        nbins=30,
        marginal="rug",
        title=f"Histogramme de {variable} selon {cible}",
        color_discrete_map={"Negative": "#4C78A8", "Positive": "#F58518"},
        opacity=0.7
    )
    return fig


def clients_boxplot(clients, variable, cible="reponse_client"):
    dataframe = clients[[variable, cible]].copy()
    dataframe[cible] = dataframe[cible].map({0: "Negative", 1: "Positive"})
    
    fig = px.box(
        dataframe,
        x=cible,
        y=variable,
        color=cible,
        title=f"Boxplot de {variable} selon {cible}",
        color_discrete_map={"Negative": "#4C78A8", "Positive": "#F58518"}
    )
    return fig


def clients_kdeplot(clients, variable, cible="reponse_client"):
    dataframe = clients[[variable, cible]].copy()
    dataframe[cible] = dataframe[cible].map({0: "Negative", 1: "Positive"})
    
    # Plotly n'a pas de KDE direct identique à Seaborn, on utilise un histogramme de densité
    fig = px.histogram(
        dataframe,
        x=variable,
        color=cible,
        marginal="violin",
        histnorm='probability density',
        title=f"Distribution (Densité) de {variable} selon {cible}",
        color_discrete_map={"Negative": "#4C78A8", "Positive": "#F58518"},
        barmode='overlay'
    )
    return fig

def clients_correlation_matrix(clients):
    numeric_df = clients.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method='spearman')
    
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1],
        title="Matrice de Corrélation de Spearman"
    )
    return fig

def clients_target_distribution(clients, cible="reponse_client"):
    counts = clients[cible].value_counts()
    fig = px.pie(
        values=counts.values,
        names=["Non Intéressé (0)", "Intéressé (1)"],
        title="Répartition de la Variable Cible",
        color_discrete_sequence=["#4C78A8", "#F58518"]
    )
    return fig

def clients_scatter_relation(clients, x, y, cible="reponse_client"):
    df_sample = clients.sample(min(5000, len(clients)), random_state=42).copy()
    df_sample[cible] = df_sample[cible].map({0: "Non Intéressé", 1: "Intéressé"})
    
    fig = px.scatter(
        df_sample,
        x=x,
        y=y,
        color=cible,
        opacity=0.5,
        title=f"Analyse croisée : {x} vs {y}",
        color_discrete_map={"Non Intéressé": "#4C78A8", "Intéressé": "#F58518"},
        labels={x: x.replace('_', ' ').title(), y: y.replace('_', ' ').title()}
    )
    
    # Ajout des lignes de moyenne
    fig.add_vline(x=clients[x].mean(), line_dash="dash", line_color="grey")
    fig.add_hline(y=clients[y].mean(), line_dash="dash", line_color="grey")
    
    return fig

def preparer_dataset_complet(df, is_train=True, mappings=None):
    """
    Transforme les données. 
    Si is_train=True : apprend les mappings et les renvoie.
    Si is_train=False : utilise les mappings fournis pour transformer.
    """
    df_prep = df.copy()
    
    # 1. Tranches d'âge
    df_prep['tranche_age'] = pd.cut(df_prep['age'], bins=7, labels=False)
    
    # 2. Encodage logique métier (Target Encoding)
    current_mappings = {}
    cols_metier = ['code_regional', 'canal_communication']
    
    for col in cols_metier:
        if is_train:
            # On calcule le taux de réponse moyen par modalité sur le train
            mapping = df.groupby(col)['reponse_client'].mean().to_dict()
            current_mappings[col] = mapping
        else:
            # On utilise le mapping passé en argument (issu du train)
            mapping = mappings[col] if mappings and col in mappings else {}
        
        df_prep[f'{col}_score'] = df_prep[col].map(mapping).fillna(0)
    
    # 3. Encodage des variables binaires et ordinales
    df_prep['genre'] = df_prep['genre'].map({'Male': 1, 'Female': 0}).fillna(0)
    df_prep['vehicule_endommage'] = df_prep['vehicule_endommage'].map({'Yes': 1, 'No': 0}).fillna(0)
    df_prep['age_vehicule'] = df_prep['age_vehicule'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}).fillna(0)
    
    # 4. Nettoyage des colonnes non prédictives
    cols_to_drop = ['id_client', 'code_regional', 'canal_communication']
    # On ne drop 'reponse_client' que s'il existe (il n'existe pas dans le fichier de prod)
    if 'reponse_client' in df_prep.columns and not is_train:
        cols_to_drop.append('reponse_client')
        
    df_prep = df_prep.drop(columns=[c for c in cols_to_drop if c in df_prep.columns])
    df_prep = df_prep.fillna(0)
    
    if is_train:
        return df_prep, current_mappings
    return df_prep

def entrainer_modele_rf(df_train_prepare):
    """
    Prend un dataframe déjà préparé et entraîne le modèle.
    """
    X = df_train_prepare.drop(columns=['reponse_client'])
    y = df_train_prepare['reponse_client']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # On initialise le scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entraînement
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test, X.columns

def generer_predictions_marketing(model, scaler, df_clients, feature_names, mappings):
    """
    Prépare les données de prod en utilisant les outils (scaler, mappings) du train.
    """
    # 1. Préparation avec les mappings du train
    df_prep = preparer_dataset_complet(df_clients, is_train=False, mappings=mappings)
    
    # 2. Alignement des colonnes (sécurité)
    X_prod = df_prep[feature_names]
    
    # 3. Scaling avec le scaler du TRAIN (Crucial !)
    X_prod_scaled = scaler.transform(X_prod)
    
    # 4. Probabilités
    probs = model.predict_proba(X_prod_scaled)[:, 1]
    
    resultats = df_clients.copy()
    resultats['probabilite_souscription'] = probs
    
    def definir_strategie(p):
        if p > 0.8: return "Presque certain (Contact inutile)"
        if p < 0.2: return "Peu probable (Ne pas contacter)"
        if 0.35 <= p <= 0.65: return "À CIBLER (Zone d'influence)"
        return "Secondaire"

    resultats['strategie_marketing'] = resultats['probabilite_souscription'].apply(definir_strategie)
    return resultats