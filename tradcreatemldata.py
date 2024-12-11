#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import argparse
from tradautotools import rmfile
from datetime import datetime
from tradparams import to_float, pseudos, minutes_in_a_year, period, testnum, data_generator_categorizing_threshold

threshold = data_generator_categorizing_threshold

def floor_the_date(date_obj):

    # Floor the date to the beginning of the year
    floored_date = datetime(year=date_obj.year, month=1, day=1)

    return floored_date


def to_minute_in_the_year(date_str):

    # Convert string to a datetime object
    date_obj = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
    floored_date_obj = floor_the_date(date_obj)
    # Convert the datetime object to Unix timestamp (seconds since epoch)
    timestamp_seconds = date_obj.timestamp()
    floored_timestamp_seconds = floored_date_obj.timestamp()

    # Convert the timestamp to minutes
    minutes_in_the_year = (timestamp_seconds - floored_timestamp_seconds) / 60 #* minutes_in_a_year

    return minutes_in_the_year

def minute_in_year_normalization(date_str):
    return 4 * (to_minute_in_the_year(date_str) - minutes_in_a_year / 2) / minutes_in_a_year


def create_custom_csv(input_csv, output_csv, period, testnum):
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
    for i in range(len(df) - period):  # Assure qu'on peut avoir 480 scores + 5 pour la somme suivante
        # Sélectionner 480 scores consécutifs
        scores_window = df['candle_score'].iloc[i:i + period - testnum].tolist()
        time_stamp = df['time'].iloc[i + period - testnum - 1]
        # Calculer la somme des 5 scores suivants
        next_five_scores = df['candle_score'].iloc[i + period - testnum:i + period]
        next_five_sum = next_five_scores.sum()

        # Vérifier la condition de différence relative (seulement pour les lignes après la première)
        #if rows:
        #    prev_sum = rows[-1][-1]  # La dernière somme des 5 scores de la ligne précédente
        #    if abs(next_five_sum - prev_sum) <= abs(0.5 * prev_sum):
        #        continue  # Passer cette ligne si la condition de différence n'est pas remplie

        # Ajouter la fenêtre des scores et la somme des 5 scores suivants
        myn = minute_in_year_normalization(time_stamp)
        if next_five_sum >= threshold:
            val = 1
        elif next_five_sum <= -threshold:
            val = -1
        else:
            val = 0
        rows.append([myn] + scores_window + [val])

    # Créer un DataFrame pour les lignes sélectionnées
    result_df = pd.DataFrame(rows)

    # Enregistrer le DataFrame dans le fichier de sortie
    result_df.to_csv(output_csv, index=False, header=False)
    print(f"Fichier {output_csv} créé avec succès avec {len(rows)} lignes.")
    return result_df


def main(period, testnum):
    for x in pseudos:
        rmfile(f"{pseudos[x]}_data.csv")
        rmfile(f"{pseudos[x]}_timestamp_data.csv")
        df = create_custom_csv(f"{pseudos[x]}_scores.csv",f"{pseudos[x]}_data.csv", period, testnum)
        print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='order placing function for metatrader 5')

    parser.add_argument(
        "-p",
        "--period",
        help="Period",
        default=period,
        type=int
    )

    parser.add_argument(
        "-t",
        "--testnum",
        help="Test number of candlesticks",
        default=testnum,
        type=int
    )

    args = parser.parse_args()
    main(period=args.period, testnum=args.testnum)
