import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def factors(n):
    
    '''
    Fonction donnant la décomposition en facteurs premiers d'un entier n.
    '''
    
    result = []
    
    for i in range(2,n+1):
        
        while n/float(i) == int(n/float(i)):
            n = n/float(i)
            result.append(i)
        
        if n == 1:
            return result

        
def get_n_rowcol(fields):
    
    '''
    Fonction définissant un nombre de lignes et colonnes pour un plot, de manière à distribuer les variables contenues dans 'field'.
    '''
    
    n_fields = len(fields)
    
    n_factors = factors(n_fields)
    
    nrow = 1
    ncol = 1

    for i in range (0,len(n_factors)):
        val = n_factors[-(i+1)]

        #Nombre de lignes doit être >= au nombre de colonnes
        if ncol*val > nrow :
            nrow = nrow*val
        else:
            ncol = ncol*val

    return nrow, ncol


def plot_top_feat_hist(top_exp_feats, hist_data, cust_X, c_col):
    
    """
    Fonction de représentation de la position d'un client au sein d'un histogramme, dont les données sont issues d'un dict ou json.
    
    Paramètres:
    -----------
    - top_exp_feats : itérable contenant les noms des variables à représenter
    - hist_data : données contenant les coordonnées des bins à représenter, pour chacune des variables retenues, format dict ou json
    - cust_X : valeur de la variable pour le client
    - c_col : couleur servant à représenter le client
    
    Résultat:
    ---------
    histogramme reprenant la distribution de la variable retenue au sein de la base client (en gris), la moyenne de celle-ci (trait pointillé noir), et la position du client (trait plein de couleur à définir)
    """
    
    fig, axes = plt.subplots(nrows = len(top_exp_feats), figsize=(10,2*len(top_exp_feats)))

    for feat, ax in zip(top_exp_feats, fig.axes):
        # Extraction des coordonnées (x) des bins
        bins = hist_data[feat]['bins']
        # Extraction des coordonnées (y) des bins
        values = hist_data[feat]['values']
        # Extraction de la moyenne
        mean = hist_data[feat]['mean']

        # Représentation de l'histogramme
        ax.bar(
            [(a+b)/2 for a,b in zip(bins[:-1],bins[1:])],
            values,
            width=bins[1]-bins[0],
            color='grey'
        )
        
        # Fixation des limites des axes
        ax.set_xlim(
            left=bins[0],
            right=bins[-1]
        )
        
        ax.set_ylim(
            bottom=ax.get_ylim()[0],
            top=ax.get_ylim()[1]
        )
        
        # Représentation de la position du client
        ax.vlines(
            x=cust_X[feat],
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            color=c_col,
            label='Client'
        )
        
        # Représentation de la moyenne
        ax.vlines(
            x=mean,
            ymin=ax.get_ylim()[0],
            ymax=ax.get_ylim()[1],
            color='black',
            linestyle='--',
            label='Moyenne'
        )
        
        ax.legend()
        
        ax.set_title(f"Distribution de l'indicateur {feat}")

    plt.tight_layout()    

    return fig