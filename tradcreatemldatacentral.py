#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import tradparams as tp

def create_custom_csv(input_csv, output_csv):
    """
    Crée un fichier CSV avec des lignes de 481 colonnes contenant 480 scores consécutifs et la somme des 5 scores suivants.
    
    :param input_csv: Nom du fichier CSV d'entrée (doit contenir les colonnes 'time' et 'candle_score').
    :param output_csv: Nom du fichier CSV de sortie.
    """
    # Charger les données d'entrée
    df = pd.read_csv(input_csv)

    # Liste pour stocker les lignes valides à ajouter au nouveau fichier CSV
    rows = []

    # Boucle pour parcourir les scores en partant du 480ème plus ancien jusqu'au 6ème plus récent
    for i in range(len(df) - 485):  # Assure qu'on peut avoir 480 scores + 5 pour la somme suivante
        # Sélectionner 480 scores consécutifs
        scores_window = df['candle_score'].iloc[i:i + 480].tolist()

        # Calculer la somme des 5 scores suivants
        next_five_scores = df['candle_score'].iloc[i + 480:i + 485]
        next_five_sum = next_five_scores.sum()

        # Vérifier la condition de différence relative (seulement pour les lignes après la première)
        if rows:
            prev_sum = rows[-1][-1]  # La dernière somme des 5 scores de la ligne précédente
            if abs(next_five_sum - prev_sum) <= abs(0.5 * prev_sum):
                continue  # Passer cette ligne si la condition de différence n'est pas remplie

        # Ajouter la fenêtre des scores et la somme des 5 scores suivants
        rows.append(scores_window + [next_five_sum])

    # Créer un DataFrame pour les lignes sélectionnées
    result_df = pd.DataFrame(rows)

    # Enregistrer le DataFrame dans le fichier de sortie
    result_df.to_csv(output_csv, index=False, header=False)
    print(f"Fichier {output_csv} créé avec succès avec {len(rows)} lignes.")
    return result_df

import os

def concatenate_csv_files(output_file):
    """
    Concatène une liste de fichiers CSV homogènes en un seul fichier CSV.

    :param file_list: Liste des chemins des fichiers CSV à concaténer.
    :param output_file: Chemin du fichier CSV de sortie.
    """
    # Liste pour stocker les DataFrames de chaque fichier CSV
    data_frames = []
    
    for x in tp.pseudos:
        # Lire chaque fichier CSV et l'ajouter à la liste des DataFrames
        df = pd.read_csv(f"{tp.pseudos[x]}_data.csv")
        data_frames.append(df)

    # Concaténer tous les DataFrames dans un seul DataFrame
    concatenated_df = pd.concat(data_frames, ignore_index=True)

    # Enregistrer le DataFrame concaténé dans le fichier de sortie
    concatenated_df.to_csv(output_file, index=False)
    print(f"Fichier {output_file} créé avec succès en concaténant les données de chaque Symbol.")


def main():
    concatenate_csv_files("data.csv")
                                                         

if __name__ == '__main__':
    main()
