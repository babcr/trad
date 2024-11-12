#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from collections import deque
from tradparams import pseudos
from tradparams import mperiod
from tradparams import mt5
from tradparams import period
from tradcreatemldata import minute_in_year_normalization
from datetime import datetime, timedelta
from time import sleep
from numpy import void
import xgboost as xgb
from so import main as so
from tradparams import dashboard
from tradparams import delta_timeframe_pair_pseudos
from tradparams import model_in_use
from tradparams import testnum
#from tradlearnXGBoost import np 
#from tradlearnXGBoost import cert_deg
#from tradlearnXGBoost import test_model

scores = {}
present_times = {}
candles = {}
base_sums = {}
base_sum_supports = {}
elements = {}
model = None

def calculate_candlestick_score_realtime(symbol, last_candle, number_of_prediction_candlesticks=mperiod, new=True):
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
        try:
            base_sum_supports[symbol].append(high_price - low_price)
            base_sums[symbol] = base_sums[symbol] + base_sum_supports[symbol][-1]
        except KeyError as ke:
            base_sum_supports[symbol] = deque([high_price - low_price])
            base_sums[symbol] = high_price - low_price
        
        
        if len(base_sum_supports[symbol]) > number_of_prediction_candlesticks:
            base_sums[symbol] = base_sums[symbol] - base_sum_supports[symbol].popleft()
    else:
        base_sums[symbol] = base_sums[symbol] - base_sum_supports[symbol][-1]
        base_sum_supports[symbol][-1] = high_price - low_price
        base_sums[symbol] = base_sums[symbol] + base_sum_supports[symbol][-1]
    base = float(base_sums[symbol] / len(base_sum_supports[symbol]))
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
    score = (body_length / base) * (1 - wick_length / total_length)
    
    return score

def load_candles(symbol, back_period=mperiod):
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

    # Télécharger les chandeliers
    cands = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, back_period)  # Exemple TIMEFRAME_D1 pour des chandeliers quotidiens

    if cands is None:
        print("Erreur lors du chargement des données de chandeliers")
        mt5.shutdown()
        return None
    
    mt5.shutdown()
    return cands

def load_candle(symbol):
    # Initialiser MetaTrader5
    if not mt5.initialize():
        print("Echec de l'initialisation de MetaTrader5")
        return None

    # Télécharger les chandeliers
    candle = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1)

    if candle is None:
        print("Erreur lors du chargement des données de chandeliers")
        mt5.shutdown()
        return None
    
    last_candle_time = None
    if symbol in candles:
        timestamp = candles[symbol][-1][0]
        if type(timestamp) == void:
            timestamp = timestamp[0]
        last_candle_time = datetime.fromtimestamp(timestamp)
        


    #date_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    # Obtenir l'heure de la bougie
    current_candle_time = datetime.fromtimestamp(candle[0][0])

    #t = minute_in_year_normalization(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Vérifier si la bougie est nouvelle ou une mise à jour
    if last_candle_time is None:
        #print("Première vérification : Bougie actuelle chargée.")
        candles[symbol] = deque(candle)
        scores[symbol] = deque(float(calculate_candlestick_score_realtime(symbol, candle)))
        present_times[symbol] = t
    elif current_candle_time > last_candle_time:
        #print(f"Nouvelle bougie détectée : {datetime.fromtimestamp(current_candle_time)}")
        candles[symbol].append(candle)
        scores[symbol].append(float(calculate_candlestick_score_realtime(symbol, candle)))
        present_times[symbol] = t
        if len(scores[symbol]) > period - testnum:
            scores[symbol].popleft()
        if len(candles[symbol]) > mperiod:
            candles[symbol].popleft()
    else:
        candles[symbol] = deque(candles[symbol])
        #print(f"Mise à jour de la bougie actuelle : {current_candle_time}")
        candles[symbol].pop()
        candles[symbol].append(candle)
        scores[symbol].pop()
        scores[symbol].append(float(calculate_candlestick_score_realtime(symbol, candle, new=False)))
        present_times[symbol] = t
    
    mt5.shutdown()
    return scores[symbol]

def load_model(model_name):
    # Load the model from the file
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_name)
    #means = test_model()
    return loaded_model

def init_env():
    global candles
    global scores
    global model
    model = load_model(model_in_use)
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for x in pseudos:
        candles[pseudos[x]] = load_candles(pseudos[x])
        scores[pseudos[x]] = deque()
        for i in range(0,period-testnum):
            scores[pseudos[x]].append(
                calculate_candlestick_score_realtime(pseudos[x], candles[pseudos[x]][mperiod - period + testnum + i - 1])
            )
            present_times[pseudos[x]] = t

def load_element(symbol):
    myn = minute_in_year_normalization(present_times[symbol])
    ele = scores[symbol].copy()
    ele.appendleft(myn)
    return [list(ele)]



def apply_model(model,element):
    dtest = xgb.DMatrix(element)
    # Faire une prédiction
    preds = model.predict(dtest)

    # Exemple de conversion des scores de prédiction (logits) en probabilités avec softmax
    '''
    logits = model.predict(dtest, output_margin=True)  # Obtenez les scores bruts
    preds_prob = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    preds_prob_mean = np.mean(preds_prob, axis=0)

    x = (preds_prob[:,0] > preds_prob_mean[0]*cert_deg) & (np.array(preds.ravel()) == 0)
    '''
    return preds


def check_open_positions_or_pending_orders(symbol):
    # Connexion à MetaTrader 5
    if not mt5.initialize():
        print("Erreur de connexion à MetaTrader 5")
        return None
    
    # Vérification des positions ouvertes pour le symbole donné
    open_positions = mt5.positions_get(symbol=symbol)
    has_open_positions = open_positions is not None and len(open_positions) > 0
    
    # Vérification des ordres en attente pour le symbole donné
    pending_orders = mt5.orders_get(symbol=symbol)
    has_pending_orders = pending_orders is not None and len(pending_orders) > 0
    
    # Déconnexion de MetaTrader 5
    mt5.shutdown()

    # Résultats
    if has_open_positions and has_pending_orders:
        return True
    elif has_open_positions:
        return True
    elif has_pending_orders:
        return True
    else:
        return False

last_orders = {}
def loop():
    for x in pseudos:
        load_candle(pseudos[x])
        ele = load_element(pseudos[x])
        pred = apply_model(model, ele)[0]
        print(f"La prediction pour {pseudos[x]} est: {pred}")
        if pseudos[x] not in last_orders:
            last_orders[pseudos[x]] = 0
        if pred == 2:
            if check_open_positions_or_pending_orders():
                if datetime.now() - last_orders[pseudos[x]] < timedelta(hour=1):
                    continue
            so(
                symbol      =   pseudos[x],
                ordertype   =   "buy_now",
                volume      =   None,
                price       =   None,
                mode        =   dashboard['defaultTradingMode'],
                delta_timeframe_pair = delta_timeframe_pair_pseudos['h']
            )
            last_orders[pseudos[x]] = datetime.now()

        elif pred == 0:
            if check_open_positions_or_pending_orders():
                if datetime.now() - last_orders[pseudos[x]] < timedelta(hour=1):
                    continue
            so(
                symbol      =   pseudos[x],
                ordertype   =   "sell_now",
                volume      =   None,
                price       =   None,
                mode        =   dashboard['defaultTradingMode'],
                delta_timeframe_pair = delta_timeframe_pair_pseudos['h']
            )
            last_orders[pseudos[x]] = datetime.now()
        else:
            pass
    sleep(60)

def main():
    init_env()
    while (True):
        loop()

if __name__ == '__main__':
    main()
    
# Utilisation du modèle chargé pour effectuer des prédictions
    #predictions = loaded_model.predict(X_test)