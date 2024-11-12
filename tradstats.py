#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import tradparams as tp
import tradautotools as ta
from collections import deque

base_sum_support = deque()
base_sum = 0

def calculate_candlestick_score(last_candle, number_of_prediction_candlesticks=24 * tp.mean_period, new=True):
    global base_sum
    global base_sum_support
    """
    Calcule le score de la dernière bougie dans un ensemble de données de bougies d'une heure sur 24h.
    
    Paramètres:
    data (pd.DataFrame): DataFrame avec les colonnes `open`, `high`, `low`, `close` pour les 24 dernières heures.
    
    Retourne:
    float: Le score de la dernière bougie.
    """
    open_price = last_candle['open']
    close_price = last_candle['close']
    high_price = last_candle['high']
    low_price = last_candle['low']

    if new:
        base_sum_support.append(high_price - low_price)
        base_sum = base_sum + base_sum_support[-1]
        if len(base_sum_support) > number_of_prediction_candlesticks:
            base_sum = base_sum - base_sum_support.popleft()
    else:
        base_sum = base_sum - base_sum_support[-1]
        base_sum_support[-1] = high_price - low_price
        base_sum = base_sum + base_sum_support[-1]
    base = base_sum / len(base_sum_support)

    print(f"base = {base}")
    # Calcul de la longueur du corps de la dernière bougie
    body_length = close_price - open_price
    
    # Calcul de la longueur des mèches
    upper_wick = high_price - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low_price
    wick_length = upper_wick + lower_wick
    
    # Calcul de la longueur totale (corps + mèches)
    total_length = abs(body_length) + wick_length
    
    # Calcul du score
    if total_length == 0:
        return 0  # Pour éviter la division par zéro si la longueur totale est nulle
    score = (body_length / base)*(1 - wick_length / total_length)
    
    return score

# Fonction principale pour lire, calculer et écrire le fichier CSV
def create_candlestick_score_csv(input_csv, output_csv):
    """
    Lit un fichier CSV de chandeliers, calcule le score de chaque chandelier et crée un nouveau CSV avec le score.

    :param input_csv: Nom du fichier CSV d'entrée.
    :param output_csv: Nom du fichier CSV de sortie.
    :param base: Base utilisée pour le calcul du score des chandeliers.
    """
    # Lire le fichier CSV d'entrée
    df = pd.read_csv(input_csv)

    # Calculer le score des chandeliers pour chaque ligne
    df['candle_score'] = df.apply(lambda row: calculate_candlestick_score(row), axis=1)

    # Sélectionner uniquement les colonnes 'time' et 'candle_score'
    result_df = df[['time', 'candle_score']]

    # Enregistrer dans un nouveau fichier CSV
    result_df.to_csv(output_csv, index=False)
    print(f"Fichier CSV {output_csv} créé avec succès.")
    return result_df


def main():
    global base_sum
    global base_sum_support
    for x in tp.pseudos:
        base_sum_support = deque()
        base_sum = 0
        ta.rmfile(f"{tp.pseudos[x]}_scores.csv")
        candles_scores_df = create_candlestick_score_csv(f"{tp.pseudos[x]}_candlesticks.csv", f"{tp.pseudos[x]}_scores.csv")
        print(candles_scores_df)


if __name__ == '__main__':

    main()