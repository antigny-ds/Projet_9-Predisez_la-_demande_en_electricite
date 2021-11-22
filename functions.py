import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from sklearn.metrics import *
from statsmodels.tsa.stattools import acf, adfuller, pacf

# =========================================================================================================================
# Fonctions : Data description
# =========================================================================================================================
def exploration_forme(dataframe):
    df = dataframe.copy()
    
    display(df.head(5))
    
    print("Nombre de lignes :", df.shape[0])
    print("Nombre de colonnes :", df.shape[1], "\n")
    
    for i in range(len(df.dtypes.value_counts())):
        print("Nombre de variables de type", df.dtypes.value_counts().index[i], ":", df.dtypes.value_counts()[i])
    
    print(" ")
    print("Pourcentage de valeurs manquantes par variable")
    print((df.isna().sum()/df.shape[0]).sort_values(ascending=True))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.isna(), cbar=False)
    plt.yticks(np.arange(0, df.shape[0], 12), df.index.year[::12])
    
    plt.show()
    
    print("Premier mois :", str(df.index[0].year) +'-' + str(df.index[0].month))
    print("Dernier mois :", str(df.index[-1].year) +'-' + str(df.index[-1].month))


def check_valeurs(table, colonne):
    
    test = [value for value in table["conso"] if re.search(r"^\d+$", str(value)) is None]
    print("Colonne", str(colonne), "\n")
    print("Valeurs qui ne suivent pas le formatage défini :\n")
    print(pd.unique(test), "\n")
    print("Nombre d'occurence par valeur :\n")
    print(pd.DataFrame(test).value_counts(), "\n")
    print("Nombre de valeur total dont le formatage est différent :", len(test))
    print("-"*100)
    

# =========================================================================================================================
# Fonctions : Tests statistiques et métriques
# =========================================================================================================================
# Test de Shapiro-Wilk
def shapiro_test(array, alpha):
    F, p = stats.shapiro(array)

    print('Résultat du test de Shapiro-Wilk')
    print(f'- Statistique de test : {F}, p_value : {p}')
    if p < alpha:
        print('- H0 rejettée, Distribution probablement non Gaussienne')
    else:
        print('- H0 validée, Distribution probablement Gaussienne')


# Test de Kolmogorov-Smirnov
def ks_test(array, alpha):
    # Test de Kolmogorov-Smirnov
    gauss = np.random.normal(loc=array.mean(),
                             scale=array.std(ddof=0),
                             size=len(array))

    F, p = stats.ks_2samp(array, gauss)

    print('\nResultat du test de Kolmogorov-Smirnov')
    print(f'- Statistique de test : {F}, p_value : {p}')
    if p < alpha:
        print("- H0 rejettée, Distribution probablement non Gaussienne")
    else:
        print("- H0 validée, Distribution probablement Gaussienne")


def adf_test(array, alpha):
    # Test Augmenté de Dickey-Fuller
    res = adfuller(array)
    
    print('Resultat du test de Dickey-Fuller')
    print(f'- ADF statistique{res[0]}, p_value : {res[1]}')
    if res[1] < alpha:
        print("- H0 rejettée, Série probablement stationnaire")
    else:
        print("- H0 validée, Série probablement instationnaire")


def reg_metrics(test, pred):
    print('Mean Absolute Percentage Error :', mean_absolute_percentage_error(test, pred))
    print('Mean Absolute Error :', mean_absolute_error(test, pred))
    print('Median absolute Error :', median_absolute_error(test, pred))
    print('Root Mean Squared Error :', np.sqrt(mean_squared_error(test, pred)))
#    print('R² :', r2_score(test, pred))


# =========================================================================================================================
# Fonctions : Graphiques
# =========================================================================================================================
# Comparaison données / modèle de régression linéaire
def plot_regression(df, model, fig_name):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.scatterplot(data=df,
                    x='DJU',
                    y='conso')

    x = np.linspace(0, df['DJU'].max(), len(df))
    y = model.coef_[0][0] * x + model.intercept_[0]
    plt.plot(x, y, color='r')

    sns.despine()

    plt.xlabel('Degré Jour Unifié')
    plt.ylabel('Consommation électrique (en GWh)')
    plt.legend(labels=['Modèle de régression : ax + b'])
    plt.tight_layout()
    
    plt.savefig(fig_name, bbox_inches='tight')

    plt.show()


# Normalité des résidus standardisés
def plot_residus_std(array, fig_name):
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.histplot(data=array, kde=True, binwidth=0.5, ax=ax)

    plt.xlabel('Résidus normalisés')
    plt.ylabel("Nombre d'enregistrements")

    sns.despine()
    plt.tight_layout()
    
    plt.savefig(fig_name, bbox_inches='tight')

    plt.show()


# Quantile-Quantile plot
def plot_qqplot(array, fig_name):
    fig, ax = plt.subplots(figsize=(8, 6))

    stats.probplot(array, plot=plt, rvalue=False)
    
    ax.set_title('Q-Q plot des résidus')
    ax.get_lines()[0].set_markerfacecolor('#1f77b4')
    ax.get_lines()[0].set_markeredgecolor('None')
    plt.xlabel('Quantiles théoriques')
    plt.ylabel('Quantiles échantillon')

    sns.despine()
    plt.tight_layout()
    
    plt.savefig(fig_name, bbox_inches='tight')
    
    plt.show()


# Comparaison 2 séries temporelles
def plot_compare(data, labels, fig_name):
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.lineplot(data=data)

    sns.despine()
    plt.xlabel('Date')
    plt.ylabel('Consommation électrique (en GWh)')
    plt.legend(labels=labels)
    plt.tight_layout()
    
    plt.savefig(fig_name, bbox_inches='tight')

    plt.show()


# Comparaison test / prédiction
def plot_test(train, test, pred, fig_name, zoom=False):
    if zoom:
        plt.figure(figsize=(6, 6))
    
        plt.plot(test, label='Test', c='orange')
        plt.plot(pred, label='Prédiction', c='green', linestyle='--')
    
        plt.xticks(rotation=30)
    
    else:
        plt.figure(figsize=(14, 6))

        plt.plot(train, label='Entraînement')
        plt.plot(test, label='Test')
        plt.plot(pred, label='Prédiction', linestyle='--')
        
    sns.despine()
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(fig_name, bbox_inches='tight')
    
    plt.show()


# Prédiction
def plot_pred(known, pred, fig_name):
    plt.figure(figsize=(14, 6))
    
    plt.plot(known, label='Série')
    plt.plot(pred, label='Prédiction', linestyle='--')

    sns.despine()
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(fig_name, bbox_inches='tight')
    
    plt.show()


# Stationnarité
def plot_sarima(serie, size, fig_name, titre=''):
    # Layout
    fig = plt.figure(constrained_layout=True, figsize=(10, 8))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)

    # Time serie
    sns.lineplot(data=serie, ax=ax1)
    ax1.set_title(titre)
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    # PACF
    serie_pacf = pacf(serie, nlags=40, method='ywm')
    ax2.stem(range(len(serie_pacf)), serie_pacf)
    ax2.set_title('Partial Autocorrelation')
    ax2.axhline(y=0, c='C0')
    ax2.axhline(y=-1.96/np.sqrt(size), c='w', linestyle='--', linewidth=0.8)
    ax2.axhline(y=1.96/np.sqrt(size), c='w', linestyle='--', linewidth=0.8)
    ax2.set_xlim([-1, 40])
    ax2.set_ylim([-1.1, 1.1])
    
    # ACF
    serie_acf = acf(serie, nlags=40, fft=False)
    ax3.stem(range(len(serie_acf)), serie_acf)
    ax3.set_title('Autocorrelation')
    ax3.axhline(y=0, c='C0')
    ax3.axhline(y=-1.96/np.sqrt(size), c='w', linestyle='--', linewidth=0.8)
    ax3.axhline(y=1.96/np.sqrt(size), c='w', linestyle='--', linewidth=0.8)
    
    plt.savefig(fig_name, bbox_inches='tight')

    plt.show()


# Comparaison des prédictions
def plot_compare_pred(serie, hw, sarima, fig_name):       
    plt.figure(figsize=(8, 6))

    plt.plot(serie, label='Consommation corrigée')
    plt.plot(hw, label='Holt-Winters', linestyle='--')
    plt.plot(sarima, label='SARIMA', linestyle='--')
        
    sns.despine()
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(fig_name, bbox_inches='tight')
    
    plt.show()
