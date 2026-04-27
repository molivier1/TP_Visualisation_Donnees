import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
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


def clients_outliers_summary(clients, variables):
    rows = []
    for variable in variables:
        q1 = clients[variable].quantile(0.25)
        q3 = clients[variable].quantile(0.75)
        iqr = q3 - q1
        borne_basse = q1 - 1.5 * iqr
        borne_haute = q3 + 1.5 * iqr
        masque = (clients[variable] < borne_basse) | (clients[variable] > borne_haute)
        nb_outliers = int(masque.sum())
        part_outliers = round(100 * nb_outliers / len(clients), 2)
        conclusion = "A surveiller" if part_outliers >= 5 else "Impact limité"
        rows.append({
            "variable": variable,
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "borne_basse": round(borne_basse, 2),
            "borne_haute": round(borne_haute, 2),
            "nb_outliers": nb_outliers,
            "part_outliers_pct": part_outliers,
            "conclusion_metier": conclusion
        })
    return pd.DataFrame(rows)


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


def clients_marketing_profile_summary(clients_cibles):
    if clients_cibles.empty:
        return pd.DataFrame(columns=["indicateur", "valeur"])

    top_genre = clients_cibles["genre"].mode().iloc[0]
    top_age_vehicule = clients_cibles["age_vehicule"].mode().iloc[0]
    top_dommage = clients_cibles["vehicule_endommage"].mode().iloc[0]

    return pd.DataFrame([
        {"indicateur": "Nombre de clients ciblés", "valeur": len(clients_cibles)},
        {"indicateur": "Âge moyen", "valeur": round(clients_cibles["age"].mean(), 1)},
        {"indicateur": "Prime annuelle moyenne", "valeur": round(clients_cibles["prime_annuelle"].mean(), 1)},
        {"indicateur": "Ancienneté moyenne", "valeur": round(clients_cibles["anciennete"].mean(), 1)},
        {"indicateur": "Genre majoritaire", "valeur": top_genre},
        {"indicateur": "Âge véhicule majoritaire", "valeur": top_age_vehicule},
        {"indicateur": "Véhicule endommagé majoritaire", "valeur": top_dommage},
    ])

def preparer_dataset_complet(df, is_train=True, artifacts=None):
    """
    Transforme les données. 
    Si is_train=True : apprend les transformations et les renvoie.
    Si is_train=False : utilise les artefacts fournis pour transformer.
    """
    df_prep = df.copy()
    current_artifacts = {} if artifacts is None else artifacts.copy()
    
    # 1. Tranches d'âge
    df_prep['tranche_age'] = pd.cut(df_prep['age'], bins=7, labels=False)
    
    # 2. Encodage logique métier (Target Encoding)
    cols_metier = ['code_regional', 'canal_communication']
    current_artifacts.setdefault("target_mappings", {})
    
    for col in cols_metier:
        if is_train:
            # On calcule le taux de réponse moyen par modalité sur le train
            mapping = df.groupby(col)['reponse_client'].mean().to_dict()
            current_artifacts["target_mappings"][col] = mapping
        else:
            mapping = current_artifacts.get("target_mappings", {}).get(col, {})
        
        df_prep[f'{col}_score'] = df_prep[col].map(mapping).fillna(0)
    
    # 3. Encodage des variables binaires et ordinales
    genre_normalise = _normalize_text_series(df_prep['genre'])
    dommage_normalise = _normalize_text_series(df_prep['vehicule_endommage'])
    age_vehicule_normalise = _normalize_text_series(df_prep['age_vehicule'])

    onehot_input = pd.DataFrame({
        "genre": genre_normalise.replace({"female": "femelle"}),
        "vehicule_endommage": dommage_normalise.replace({"yes": "oui", "non": "no"})
    })

    if is_train:
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        onehot_matrix = onehot_encoder.fit_transform(onehot_input)
        current_artifacts["onehot_encoder"] = onehot_encoder
        current_artifacts["onehot_columns"] = onehot_encoder.get_feature_names_out(onehot_input.columns).tolist()
    else:
        onehot_encoder = current_artifacts["onehot_encoder"]
        onehot_matrix = onehot_encoder.transform(onehot_input)

    onehot_df = pd.DataFrame(
        onehot_matrix,
        columns=current_artifacts["onehot_columns"],
        index=df_prep.index
    )

    age_vehicule_series = age_vehicule_normalise.replace({
        '< 1 year': '< 1 an',
        '1-2 year': '1-2 an',
        '1-2 years': '1-2 an',
        '> 2 year': '> 2 ans',
        '> 2 years': '> 2 ans'
    })

    if is_train:
        ordinal_encoder = OrdinalEncoder(
            categories=[['< 1 an', '1-2 an', '> 2 ans']],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        age_vehicule_encoded = ordinal_encoder.fit_transform(age_vehicule_series.to_frame())
        current_artifacts["ordinal_encoder"] = ordinal_encoder
    else:
        ordinal_encoder = current_artifacts["ordinal_encoder"]
        age_vehicule_encoded = ordinal_encoder.transform(age_vehicule_series.to_frame())

    df_prep['age_vehicule'] = age_vehicule_encoded.ravel()
    df_prep = pd.concat([df_prep, onehot_df], axis=1)
    
    # 3.bis Création de variables d'interaction (Demandé dans le sujet)
    # On combine la tranche d'âge avec l'état du véhicule et l'ancienneté d'assurance
    dommage_col = 'vehicule_endommage_oui' if 'vehicule_endommage_oui' in df_prep.columns else 'vehicule_endommage_yes'
    df_prep['inter_age_dommage'] = df_prep['tranche_age'] * df_prep.get(dommage_col, 0)
    df_prep['inter_age_ancien_assure'] = df_prep['tranche_age'] * df_prep.get('ancien_assure', 0)
    df_prep['inter_vehicule_ancien'] = df_prep['age_vehicule'] * df_prep.get(dommage_col, 0)

    # 4. Nettoyage des colonnes non prédictives
    cols_to_drop = ['id_client', 'code_regional', 'canal_communication', 'genre', 'vehicule_endommage']
    # On ne drop 'reponse_client' que s'il existe (il n'existe pas dans le fichier de prod)
    if 'reponse_client' in df_prep.columns and not is_train:
        cols_to_drop.append('reponse_client')
        
    df_prep = df_prep.drop(columns=[c for c in cols_to_drop if c in df_prep.columns])
    df_prep = df_prep.fillna(0)
    
    if is_train:
        return df_prep, current_artifacts
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
    
    # Recherche légère d'hyperparamètres sur un sous-échantillon pour garder une app fluide
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [8, 12],
        'min_samples_leaf': [1, 3]
    }
    X_tune, _, y_tune, _ = train_test_split(
        X_train_scaled,
        y_train,
        train_size=0.25,
        random_state=42,
        stratify=y_train
    )
    base_model = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=1)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring='f1',
        cv=3,
        n_jobs=1
    )
    grid.fit(X_tune, y_tune)
    model = grid.best_estimator_
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X_test_scaled, y_test, X.columns, grid.best_params_, grid.best_score_

def generer_predictions_marketing(model, scaler, df_clients, feature_names, artifacts):
    """
    Prépare les données de prod en utilisant les outils (scaler, mappings) du train.
    """
    # 1. Préparation avec les mappings du train
    df_prep = preparer_dataset_complet(df_clients, is_train=False, artifacts=artifacts)
    
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
