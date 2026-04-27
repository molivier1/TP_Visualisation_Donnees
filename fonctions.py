import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc


def clients_shape(clients):
    return clients.shape

def clients_type(clients):
    return clients.dtypes

def clients_missing_values_summary(clients):
    return clients.isnull().sum()

def clients_duplicated_rows_count(clients):
    return clients.duplicated().sum()


def _normalize_text_series(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

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
        # marginal="rug" supprimé car trop lourd pour les grands datasets
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
        # marginal="violin" supprimé pour améliorer la fluidité
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
    counts = clients[cible].value_counts().sort_index()
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


def clients_confusion_matrix_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Prédit 0", "Prédit 1"],
        y=["Réel 0", "Réel 1"],
        color_continuous_scale="Blues",
        title="Matrice de confusion"
    )
    fig.update_layout(xaxis_title="Prédiction", yaxis_title="Valeur réelle")
    return fig


def clients_roc_curve_figure(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC AUC = {roc_auc:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Référence", line=dict(dash="dash")))
    fig.update_layout(
        title="Courbe ROC",
        xaxis_title="Taux de faux positifs",
        yaxis_title="Taux de vrais positifs"
    )
    return fig


def clients_pr_curve_figure(y_true, y_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=f"PR AUC = {pr_auc:.3f}"))
    fig.update_layout(
        title="Courbe Precision-Recall",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )
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
    genre_normalise = _normalize_text_series(df_prep['genre'])
    dommage_normalise = _normalize_text_series(df_prep['vehicule_endommage'])
    age_vehicule_normalise = _normalize_text_series(df_prep['age_vehicule'])

    df_prep['genre'] = genre_normalise.map({
        'male': 1,
        'm': 1,
        'homme': 1,
        'female': 0,
        'femelle': 0,
        'f': 0
    }).fillna(0)
    df_prep['vehicule_endommage'] = dommage_normalise.map({
        'yes': 1,
        'oui': 1,
        'true': 1,
        '1': 1,
        'no': 0,
        'non': 0,
        'false': 0,
        '0': 0
    }).fillna(0)
    df_prep['age_vehicule'] = age_vehicule_normalise.map({
        '< 1 year': 0,
        '< 1 an': 0,
        '1-2 year': 1,
        '1-2 years': 1,
        '1-2 an': 1,
        '1-2 ans': 1,
        '> 2 years': 2,
        '> 2 year': 2,
        '> 2 ans': 2
    }).fillna(0)
    
    # 3.bis Création de variables d'interaction (Demandé dans le sujet)
    # On combine la tranche d'âge avec l'état du véhicule et l'ancienneté d'assurance
    df_prep['inter_age_dommage'] = df_prep['tranche_age'] * df_prep['vehicule_endommage']
    df_prep['inter_age_ancien_assure'] = df_prep['tranche_age'] * df_prep.get('ancien_assure', 0)
    df_prep['inter_vehicule_ancien'] = df_prep['age_vehicule'] * df_prep['vehicule_endommage']

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
    
    # Stratégie de mise à l'échelle différenciée (Demandé dans le sujet)
    # MinMaxScaler: age, permis_conduire / RobustScaler: prime_annuelle / StandardScaler: reste
    cols_minmax = ['age', 'permis_conduire'] if 'permis_conduire' in X.columns else ['age']
    cols_robust = ['prime_annuelle']
    cols_standard = [c for c in X.columns if c not in cols_minmax + cols_robust]

    scaler = ColumnTransformer([
        ('minmax', MinMaxScaler(), cols_minmax),
        ('robust', RobustScaler(), cols_robust),
        ('std', StandardScaler(), cols_standard)
    ])

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
