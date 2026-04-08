import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def clients_shape(clients):
    return clients.shape

def clients_type(clients):
    return clients.dtypes

def clients_valeur_manquantes(clients):
    return clients[clients.isnull().sum(axis=1) > 0].copy()

def clients_duplicated_values(clients):
    return clients[clients.duplicated()]

def clients_countplot(clients, variable, cible="reponse_client"):
    dataframe = clients[[variable, cible]].copy()
    dataframe[variable] = dataframe[variable].fillna("Manquant")
    dataframe[cible] = dataframe[cible].map({0: "Negative", 1: "Positive"})

    fig, ax = plt.subplots(figsize=(8, 5))
    ordre = dataframe[variable].value_counts().index.tolist()

    sns.countplot(
        data=dataframe,
        x=variable,
        hue=cible,
        order=ordre,
        palette=["#4C78A8", "#F58518"],
        ax=ax,
    )

    total = len(dataframe)
    for barre in ax.patches:
        hauteur = barre.get_height()
        if np.isnan(hauteur) or hauteur <= 0:
            continue

        pourcentage = (hauteur / total) * 100
        x_position = barre.get_x() + (barre.get_width() / 2)
        ax.annotate(
            f"{pourcentage:.1f}%",
            (x_position, hauteur),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )

    ax.set_title(f"Distribution de {variable} selon {cible}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Nombre de clients")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Reponse client")
    fig.tight_layout()

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

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=dataframe,
        x=variable,
        hue=cible,
        kde=False,
        bins=30,
        palette=["#4C78A8", "#F58518"],
        alpha=0.6,
        ax=ax,
    )

    ax.set_title(f"Histogramme de {variable} selon {cible}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Nombre de clients")
    fig.tight_layout()

    return fig


def clients_boxplot(clients, variable, cible="reponse_client"):
    dataframe = clients[[variable, cible]].copy()
    dataframe[cible] = dataframe[cible].map({0: "Negative", 1: "Positive"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=dataframe,
        x=cible,
        y=variable,
        hue=cible,
        palette=["#4C78A8", "#F58518"],
        ax=ax,
        legend=False,
    )

    ax.set_title(f"Boxplot de {variable} selon {cible}")
    ax.set_xlabel("Reponse client")
    ax.set_ylabel(variable)
    fig.tight_layout()

    return fig


def clients_kdeplot(clients, variable, cible="reponse_client"):
    dataframe = clients[[variable, cible]].copy()
    dataframe[cible] = dataframe[cible].map({0: "Negative", 1: "Positive"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(
        data=dataframe,
        x=variable,
        hue=cible,
        fill=True,
        common_norm=False,
        palette=["#4C78A8", "#F58518"],
        alpha=0.4,
        ax=ax,
    )

    ax.set_title(f"Courbe de densite de {variable} selon {cible}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Densite")
    fig.tight_layout()

    return fig
