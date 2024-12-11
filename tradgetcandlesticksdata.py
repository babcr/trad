#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime
import argparse

import tradparams as tp


def load_candles(symbol, year_start):
    """
    Charge les chandeliers d'un symbole donné à partir du début d'une année spécifiée jusqu'à aujourd'hui.

    :param symbol: Symbole de l'instrument financier (exemple: "EURUSD").
    :param year_start: Année de début au format AAAA (par exemple, 2023).
    :return: DataFrame Pandas avec les données de chandeliers.
    """
    # Initialiser MetaTrader5
    if not mt5.initialize():
        print("Echec de l'initialisation de MetaTrader5")
        return None

    # Définir la date de début et de fin
    start_date = datetime(year_start, 1, 1)
    end_date = datetime.now()

    # Télécharger les chandeliers
    candles = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)  # Exemple TIMEFRAME_D1 pour des chandeliers quotidiens
    if candles is None:
        print("Erreur lors du chargement des données de chandeliers")
        mt5.shutdown()
        return None

    # Convertir en DataFrame pour plus de flexibilité
    candles_df = pd.DataFrame(candles)
    candles_df['time'] = pd.to_datetime(candles_df['time'], unit='s')

    # Arrêter MetaTrader5
    mt5.shutdown()

    return candles_df

def main(start_year):
    for x in tp.pseudos:
        candles_df = load_candles(tp.pseudos[x], start_year)
        if candles_df is not None:
            # Sauvegarder dans un fichier CSV si besoin
            candles_df.to_csv(f'{tp.pseudos[x]}_candlesticks.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='order placing function for metatrader 5')

    parser.add_argument(
        "-y",
        "--year", 
        help="the start year", 
        type=int
    )

    args = parser.parse_args()

    main(args.year)
