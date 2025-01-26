#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import argparse
from tradautotools import delete_files_in_directory
from datetime import datetime
from tradparams import to_float, testnum_to_gain_ratio, testnum_inter, testnum_bulk, testnum_short, testnum_wide, testnum_narrow, pseudos, period, minutes_in_a_year


def find_division_points(lst, gain):
    # Tri de la liste
    sorted_lst = sorted(lst)
    print(f"Starting elements: {sorted_lst[0:5]}")
    print(f"Ending elements: {sorted_lst[-5:]}")
    percentile = 2
    # Longueur de la liste
    n = len(sorted_lst)
    while True:
        # Calcul des indices pour les points de division
        idx1 = int(n / percentile)  # Indice du 33.33ème percentile
        idx2 = int(n * (1 - 1 / percentile))  # Indice du 66.67ème percentile

        # Points de division
        division_point1 = sorted_lst[idx1]
        division_point2 = sorted_lst[idx2]

        if division_point2 >= gain:
            break
        else:
            percentile += 1

    return division_point1, division_point2, percentile


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
    minutes_in_the_year = (timestamp_seconds - floored_timestamp_seconds) / 60 #* tp.minutes_in_a_year
    return minutes_in_the_year

def minute_in_year_normalization(date_str):
    return to_float(4 * (to_minute_in_the_year(date_str) - minutes_in_a_year / 2) / minutes_in_a_year)

def create_label_list(input_csv, period, testnum):
    # Charger les données d'entrée
    df = pd.read_csv(input_csv)

    # Liste pour stocker les lignes valides à ajouter au nouveau fichier CSV
    rows = []

    # Boucle pour parcourir les scores en partant du 480ème plus ancien jusqu'au 6ème plus récent
    for i in range(len(df) - period):  # Assure qu'on peut avoir 480 scores + 5 pour la somme suivante
        # Calculer la somme des 5 scores suivants
        next_five_scores = df['candle_score'].iloc[i + period - testnum:i + period]
        next_five_sum = next_five_scores.sum()

        # Vérifier la condition de différence relative (seulement pour les lignes après la première)
        #if rows:
        #    prev_sum = rows[-1][-1]  # La dernière somme des 5 scores de la ligne précédente
        #    if abs(next_five_sum - prev_sum) <= abs(0.5 * prev_sum):
        #        continue  # Passer cette ligne si la condition de différence n'est pas remplie

        # Ajouter la fenêtre des scores et la somme des 5 scores suivants
        rows.append(next_five_sum)
    return list(rows)

def frontiers(period, testnum, t2g=testnum_to_gain_ratio):
    llist = []
    for x in pseudos:
        llist = llist+create_label_list(f"raw_data/{pseudos[x]}_scores.csv", period, testnum)

    front = find_division_points(llist, testnum * t2g)

    print(f"Les points frontiere sont : {front}")

    return front

def create_custom_csv(input_csv, output_bull_csv, output_bear_csv, period, testnum, front):
    """
    Crée un fichier CSV avec des lignes de 481 colonnes contenant 480 scores consécutifs et la somme des 5 scores suivants.

    :param input_csv: Nom du fichier CSV d'entrée (doit contenir les colonnes 'time' et 'candle_score').
    :param output_csv: Nom du fichier CSV de sortie.
    """
    # Charger les données d'entrée
    df = pd.read_csv(input_csv)

    # Liste pour stocker les lignes valides à ajouter au nouveau fichier CSV
    rows_bull = []
    rows_bear = []
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
        
        if next_five_sum > front[1]:
            val_bull = 1
        elif next_five_sum <= front[1]:
            val_bull = 0
        else:
            continue
        

        if next_five_sum < -front[1]:
            val_bear = 1
        elif next_five_sum >= -front[1]:
            val_bear = 0
        else:
            continue

        rows_bull.append([myn] + scores_window + [val_bull])
        rows_bear.append([myn] + scores_window + [val_bear])

    # Créer un DataFrame pour les lignes sélectionnées
    result_df_bull = pd.DataFrame(rows_bull)
    result_df_bear = pd.DataFrame(rows_bear)

    # Enregistrer le DataFrame dans le fichier de sortie
    result_df_bull.to_csv(output_bull_csv, index=False, header=False)
    print(f"Fichier {output_bull_csv} créé avec succès avec {len(rows_bull)} lignes.")
    result_df_bear.to_csv(output_bear_csv, index=False, header=False)
    print(f"Fichier {output_bear_csv} créé avec succès avec {len(rows_bear)} lignes.")


def main(
        period, 
        testnum_wide=testnum_wide, 
        testnum_narrow=testnum_narrow, 
        testnum_short=testnum_short, 
        testnum_inter=testnum_inter, 
        testnum_bulk=testnum_bulk
    ):
    front_wide   = frontiers(period, testnum_wide  )
    front_bulk   = frontiers(period, testnum_bulk  )
    front_narrow = frontiers(period, testnum_narrow)
    front_inter  = frontiers(period, testnum_inter )
    front_short  = frontiers(period, testnum_short )
    for x in pseudos:
        delete_files_in_directory("bull_data/wide/")
        delete_files_in_directory("bull_data/narrow/")
        delete_files_in_directory("bull_data/short/")
        delete_files_in_directory("bull_data/inter/")
        delete_files_in_directory("bull_data/bulk/")

        delete_files_in_directory("bear_data/wide/")
        delete_files_in_directory("bear_data/narrow/")
        delete_files_in_directory("bear_data/short/")
        delete_files_in_directory("bear_data/inter/")
        delete_files_in_directory("bear_data/bulk/")

        create_custom_csv(f"raw_data/{pseudos[x]}_scores.csv",f"bull_data/wide/{pseudos[x]}_{period}_{testnum_wide}_{front_wide[2]}_data.csv", f"bear_data/wide/{pseudos[x]}_{period}_{testnum_wide}_{front_wide[2]}_data.csv", period + testnum_wide, testnum_wide, front_wide)
        create_custom_csv(f"raw_data/{pseudos[x]}_scores.csv",f"bull_data/narrow/{pseudos[x]}_{period}_{testnum_narrow}_{front_narrow[2]}_data.csv", f"bear_data/narrow/{pseudos[x]}_{period}_{testnum_narrow}_{front_narrow[2]}_data.csv", period + testnum_narrow, testnum_narrow, front_narrow)
        create_custom_csv(f"raw_data/{pseudos[x]}_scores.csv",f"bull_data/short/{pseudos[x]}_{period}_{testnum_short}_{front_short[2]}_data.csv", f"bear_data/short/{pseudos[x]}_{period}_{testnum_short}_{front_short[2]}_data.csv", period + testnum_short, testnum_short, front_short)
        create_custom_csv(f"raw_data/{pseudos[x]}_scores.csv",f"bull_data/inter/{pseudos[x]}_{period}_{testnum_inter}_{front_inter[2]}_data.csv", f"bear_data/inter/{pseudos[x]}_{period}_{testnum_inter}_{front_inter[2]}_data.csv", period + testnum_inter, testnum_inter, front_inter)
        create_custom_csv(f"raw_data/{pseudos[x]}_scores.csv",f"bull_data/bulk/{pseudos[x]}_{period}_{testnum_bulk}_{front_bulk[2]}_data.csv", f"bear_data/bulk/{pseudos[x]}_{period}_{testnum_bulk}_{front_bulk[2]}_data.csv", period + testnum_bulk, testnum_bulk, front_bulk)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='order placing function for metatrader 5') 

    parser.add_argument(
        "-p",
        "--period",
        help="Period",
        default=period,
        type=int
    )

    args = parser.parse_args()
    main(period=args.period)
