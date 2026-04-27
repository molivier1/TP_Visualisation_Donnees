# Rapport synthétique

## Contexte

Ce projet vise à prédire la probabilité de souscription d'un client afin d'optimiser une campagne marketing. L'objectif métier est de mieux cibler les clients à contacter en évitant à la fois les contacts inutiles et les prospects très peu susceptibles de convertir.

## Données utilisées

- `Data/train_info.csv` : clients avec variable cible `reponse_client`
- `Data/clients_a_contacter.csv` : clients sans cible, destinés à la phase de scoring

Variables principales :

- Variables client : `genre`, `age`, `permis_conduire`, `ancien_assure`, `anciennete`
- Variables véhicule : `age_vehicule`, `vehicule_endommage`
- Variables de contexte : `code_regional`, `canal_communication`
- Variable cible : `reponse_client`

## Exploration des données

Observations de base sur `train_info.csv` :

- 381109 lignes
- 12 colonnes
- 0 valeur manquante
- 0 doublon exact
- Jeu déséquilibré : la classe positive est minoritaire

Analyses réalisées :

- Typologie des variables
- Distribution des variables catégorielles et quantitatives
- Analyse de la variable cible
- Corrélation de Spearman
- Analyse croisée entre variables quantitatives

## Préparation des données

Transformations mises en place :

- Discrétisation de `age` en `tranche_age`
- Encodage métier de `code_regional` et `canal_communication` à partir du taux moyen de réponse
- Encodage numérique des variables catégorielles et ordinales
- Création de variables d'interaction
- Mise à l'échelle différenciée avec `MinMaxScaler`, `RobustScaler` et `StandardScaler`

Point de vigilance corrigé :

- Les modalités textuelles du dataset utilisent des valeurs comme `male`, `femelle`, `oui`, `no`, `< 1 an`, `1-2 an`, `> 2 ans`
- Le pipeline a été ajusté pour encoder correctement ces modalités

## Modélisation

Modèle retenu :

- `RandomForestClassifier` avec prise en compte du déséquilibre via `class_weight='balanced'`

Évaluation mise en avant :

- `classification_report`
- `F1-score`
- `ROC-AUC`
- `PR-AUC`
- Matrice de confusion
- Courbes ROC et Precision-Recall

## Exploitation métier

Les clients sont segmentés selon leur probabilité prédite :

- `Presque certain (Contact inutile)`
- `Peu probable (Ne pas contacter)`
- `À CIBLER (Zone d'influence)`
- `Secondaire`

La stratégie prioritaire consiste à cibler la zone intermédiaire, c'est-à-dire les profils pour lesquels l'action commerciale peut réellement influencer la décision.

## Livrables

Le dépôt contient maintenant :

- une application Streamlit : `app.py`
- un module de fonctions : `fonctions.py`
- un notebook d'analyse : `analyse_modele.ipynb`
- ce rapport synthétique : `rapport_synthetique.md`
