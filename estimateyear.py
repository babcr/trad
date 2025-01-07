from tradautotools import calculate_start_year, init_metatrader_connexion, close_metatrader_connexion
from tradparams import symbols_list, delta_timeframe_pair_pseudos, dashboard, mperiod
import argparse

# Exemple d'utilisation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='order placing function for metatrader 5')

    parser.add_argument(
        "-d",
        "--deltatimeframepair",
        help="Pair containing the period in with candle sticks statistics are calculated in hours and timeframe of the candle sticks",
        default = dashboard['defaultDeltaTimeframePair'],
        type=str,
        choices = delta_timeframe_pair_pseudos
    )

    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-n",
        "--numberofvectors",
        help="The number of vectors needed per symbols for the training",
        default=10000,
        type=int
    )

    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-p",
        "--pointspervector",
        help="The size of each vector",
        default=mperiod,
        type=int
    )

    args = parser.parse_args()
    # Initialiser MetaTrader 5
    init_metatrader_connexion()

    start_year = calculate_start_year(
        delta_timeframe_pair_pseudos[args.deltatimeframepair][1], 
        args.numberofvectors, 
        args.pointspervector
    )

    # Fermer la connexion MetaTrader 5
    close_metatrader_connexion

    print(f"Les données doivent être chargées à partir de l'année : {start_year}")