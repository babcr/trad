#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from collections import deque
from tradcreatemldata import minute_in_year_normalization
from datetime import datetime, timedelta
from time import sleep
from numpy import void, float64
from so import main as so
from tradautotools import init_metatrader_connexion, close_metatrader_connexion
from tradparams import order_types_, period, mperiod, pseudos, pseudos_we, mt5
from tradparams import limit_correlation, dashboard, delta_timeframe_pair_pseudos
from tradparams import unfilled_order_lifespan_min, hours_before_repeat_order, model_in_use_inter_bull, model_in_use_inter_bear, model_in_use_bulk_bull, model_in_use_bulk_bear, model_in_use_narrow_bull, model_in_use_narrow_bear, model_in_use_wide_bear, model_in_use_wide_bull, model_in_use_short_bear, model_in_use_short_bull
import xgboost as xgb
from numpy import ndarray,corrcoef

maxtry = 10
scores = {}
present_times = {}
candles = {}
base_sums = {}
base_sum_supports = {}
elements = {}
last_orders = {}
start_of_friday = None
end_of_sunday = None
model_bull_wide = None
model_bear_wide = None
model_bull_narrow = None
model_bear_narrow = None
model_bull_short =  None
model_bear_short =  None
model_bull_inter =  None
model_bear_inter =  None
model_bull_bulk =  None
model_bear_bulk =  None
iteration_time = datetime.now()
correlations = []


def calculate_candlestick_score_realtime(symbol, last_candle, meanperiod=mperiod, new=True):
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
            base_sums[symbol] += base_sum_supports[symbol][-1]
        except KeyError:
            base_sum_supports[symbol] = deque([high_price - low_price])
            base_sums[symbol] = float64(base_sum_supports[symbol][-1])


        if len(base_sum_supports[symbol]) > meanperiod:
            base_sums[symbol] -= base_sum_supports[symbol].popleft()
    else:
        base_sums[symbol] -= base_sum_supports[symbol][-1]
        base_sum_supports[symbol][-1] = high_price - low_price
        base_sums[symbol] += base_sum_supports[symbol][-1]

    if type(base_sums[symbol]) == ndarray:
        if base_sums[symbol].ndim == 1:
            base_sums[symbol] = base_sums[symbol][0]
    base = base_sums[symbol] / len(base_sum_supports[symbol])
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
    score = body_length * abs(body_length) / total_length / base
    if type(score) == ndarray:
        if score.ndim == 1:
            score = score[0]
    return score

def load_candles(symbol, back_period=mperiod):
    """
    Charge les chandeliers d'un symbole donné à partir du début d'une année spécifiée jusqu'à aujourd'hui.

    :param symbol: Symbole de l'instrument financier (exemple: "EURUSD").
    :param year_start: Année de début au format AAAA (par exemple, 2023).
    :return: DataFrame Pandas avec les données de chandeliers.
    """
    # Initialiser MetaTrader5
    init_metatrader_connexion()
    ok = False
    cands = None
    k = 0
    while not ok:
        # Télécharger les chandeliers
        cands = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, back_period)  # Exemple TIMEFRAME_D1 pour des chandeliers quotidiens

        if cands is not None:
            if type(cands) == ndarray:
                ok = True
        if not ok:
            print(f'Error fetching {symbol}')
            print(f'mt5 errorcode : {mt5.last_error()}')
            sleep(3)
            k =+ 1
            if k == maxtry:
                print(f"Max candles fetch try reached for {symbol}")
                close_metatrader_connexion()
                return None

    close_metatrader_connexion()
    return cands


def load_candle(symbol):
    # Initialiser MetaTrader5
    init_metatrader_connexion()

    ok = False
    k = 0
    candle = None
    while not ok:
        # Télécharger les chandeliers
        candle = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 1)
        if candle is not None:
            if type(candle) == ndarray:
                ok = True
        if not ok:
            print(f'Error fetching last candle for {symbol}')
            print(f'mt5 errorcode : {mt5.last_error()}')
            sleep(3)
            k =+ 1
            if k == maxtry:
                print(f"Max last candle fetch try reached for {symbol}")
                close_metatrader_connexion()
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
        scores[symbol] = deque(calculate_candlestick_score_realtime(symbol, candle))
        present_times[symbol] = t
    elif current_candle_time > last_candle_time:
        #print(f"Nouvelle bougie détectée : {datetime.fromtimestamp(current_candle_time)}")
        if type(candles[symbol]) == ndarray:
            candles[symbol] = deque(candles[symbol])
        candles[symbol].append(candle)
        scores[symbol].append(calculate_candlestick_score_realtime(symbol, candle))
        present_times[symbol] = t
        if len(scores[symbol]) > period:
            scores[symbol].popleft()
        if len(candles[symbol]) > mperiod:
            candles[symbol].popleft()
    else:
        candles[symbol] = deque(candles[symbol])
        #print(f"Mise à jour de la bougie actuelle : {current_candle_time}")
        candles[symbol].pop()
        candles[symbol].append(candle)
        scores[symbol].pop()
        new = calculate_candlestick_score_realtime(symbol, candle, new=False)
        scores[symbol].append(new)
        present_times[symbol] = t

    close_metatrader_connexion()
    return scores[symbol]

def load_model(model_name):
    # Load the model from the file
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_name)
    return loaded_model

def init_models():
    #global model_bull_wide
    #global model_bear_wide
    global model_bull_narrow
    global model_bear_narrow
    global model_bull_short
    global model_bear_short
    global model_bull_inter
    global model_bear_inter
    global model_bull_bulk
    global model_bear_bulk
    #model_bull_wide = load_model(model_in_use_wide_bull)
    #model_bear_wide = load_model(model_in_use_wide_bear)
    model_bull_narrow = load_model(model_in_use_narrow_bull)
    model_bear_narrow = load_model(model_in_use_narrow_bear)
    model_bull_short  = load_model(model_in_use_short_bull)
    model_bear_short  = load_model(model_in_use_short_bear)
    model_bull_inter  = load_model(model_in_use_inter_bull)
    model_bear_inter  = load_model(model_in_use_inter_bear)
    model_bull_bulk   = load_model(model_in_use_bulk_bull)
    model_bear_bulk   = load_model(model_in_use_bulk_bear)

def init_env():
    global candles
    global scores
    scores = {}
    candles = {}

    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for x in pseudos:
        candles[pseudos[x]] = load_candles(pseudos[x])
        scores[pseudos[x]] = deque()
        for i in range(0, period):
            scores[pseudos[x]].append(
                calculate_candlestick_score_realtime(pseudos[x], candles[pseudos[x]][mperiod - period + i])
            )
            present_times[pseudos[x]] = t

def load_element(symbol):
    myn = minute_in_year_normalization(present_times[symbol])
    ele = scores[symbol].copy()
    ele.appendleft(myn)
    #print(f"ele size = {len(ele)}")
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

def get_pred(bull_pred, bear_pred, scale, keyword):
    if   bull_pred > dashboard[f'bull_binary_{scale}_{keyword}'] and bear_pred <= dashboard[f'bear_binary_{scale}_{keyword}']:
        pred = 2
    elif bear_pred > dashboard[f'bear_binary_{scale}_{keyword}'] and bull_pred <= dashboard[f'bull_binary_{scale}_{keyword}']:
        pred = 0
    else:
        pred = 1
    return pred

def check_pending_orders(symbol):
    # Connexion à MetaTrader 5
    init_metatrader_connexion()

    # Vérification des ordres en attente pour le symbole donné
    pending_orders = mt5.orders_get(symbol=symbol)
    has_pending_orders = pending_orders is not None and len(pending_orders) > 0

    # Déconnexion de MetaTrader 5
    close_metatrader_connexion()

    # Résultats
    if has_pending_orders:
        return True
    else:
        return False

def check_open_positions(symbol):
    # Connexion à MetaTrader 5
    init_metatrader_connexion()

    # Vérification des positions ouvertes pour le symbole donné
    open_positions = mt5.positions_get(symbol=symbol)
    has_open_positions = open_positions is not None and len(open_positions) > 0

    # Vérification des ordres en attente pour le symbole donné
    #pending_orders = mt5.orders_get(symbol=symbol)
    #has_pending_orders = pending_orders is not None and len(pending_orders) > 0

    # Déconnexion de MetaTrader 5
    close_metatrader_connexion()

    # Résultats
    #if has_open_positions and has_pending_orders:
        #return True
    if has_open_positions:
        return True
    #elif has_pending_orders:
    #    return True
    else:
        return False

import MetaTrader5 as mt5

def cancel_unfilled_orders(symbol):
    """
    Cancel all pending orders under a given symbol that have not been filled.

    Parameters:
    symbol (str): The trading symbol (e.g., 'EURUSD', 'CADCHF').

    Returns:
    None
    """
    # Initialize the MetaTrader 5 connection
    init_metatrader_connexion()

    try:
        # Get all orders
        orders = mt5.orders_get(symbol=symbol)
        if orders is None:
            print(f"No orders found for symbol: {symbol}")
            return

        print(f"Found {len(orders)} orders for symbol {symbol}. Checking for unfilled orders...")

        # Loop through orders and cancel those that are pending (not filled)
        for order in orders:
            if order.type in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT,
                              mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP):
                # Cancel the order
                cancel_request = {
                    "action": mt5.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                }
                result = mt5.order_send(cancel_request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"Successfully canceled order {order.ticket} ({symbol})")
                else:
                    print(f"Failed to cancel order {order.ticket}, error: {result.retcode}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Shutdown the MT5 connection
        close_metatrader_connexion()


def is_weekend():
    global start_of_friday, end_of_sunday
    # Get the current time
    now = datetime.now()

    # Find the start and end of the weekend
    if start_of_friday is None or end_of_sunday is None or now - start_of_friday > timedelta(days=7):
        # Calculate the upcoming Friday and Sunday bounds
        today = now.date()
        start_of_friday = datetime.combine(today - timedelta(days=(today.weekday()-4)), datetime.min.time()) + timedelta(hours=11, minutes=59, seconds=59)
        end_of_sunday = datetime.combine(today + timedelta(days=(6 - today.weekday())), datetime.min.time()) + timedelta(hours=22, minutes=59, seconds=59)

    #print(f'start_of_friday = {start_of_friday.strftime("%Y-%m-%d %H:%M:%S")}')
    #print(f'end_of_sunday = {end_of_sunday.strftime("%Y-%m-%d %H:%M:%S")}')
    return start_of_friday <= now < end_of_sunday

def corr(v,w):
    # Calculate the Pearson correlation coefficient
    correlation_matrix = corrcoef(v, w)
    correlation_coefficient = correlation_matrix[0, 1]
    return correlation_coefficient

def get_active_symbols():
    """
    Returns a list of symbols with open positions or open orders in the MetaTrader 5 account.

    Returns:
        List[str]: A list of symbol names.
    """
    # Initialize MetaTrader 5 connection
    init_metatrader_connexion()
    # Get all open positions
    positions = mt5.positions_get()
    if positions is None:
        raise RuntimeError("Failed to get positions, error: " + mt5.last_error())

    # Get all open orders
    orders = mt5.orders_get()
    if orders is None:
        raise RuntimeError("Failed to get orders, error: " + mt5.last_error())

    # Collect symbols from positions and orders
    symbols = set()

    for position in positions:
        symbols.add(position.symbol)

    for order in orders:
        symbols.add(order.symbol)

    # Shutdown MetaTrader 5 connection
    close_metatrader_connexion()

    # Return the unique symbols as a list
    return list(symbols)

def correlated(symbol, ele):
    for op in get_active_symbols():
        if op != symbol:
            if corr(ele, load_element(op)) > limit_correlation:
                print(f"correlation {op} and {symbol} ==> {corr(ele, load_element(op))}")
                return True
    return False

def loop():
    global iteration_time
    if datetime.now() - iteration_time > timedelta(hours=1):
        init_env()
    pse = None
    if is_weekend():
        pse = pseudos_we
    else:
        pse = pseudos

    for x in pse:
        load_candle(pseudos[x])
        ele = load_element(pseudos[x])
        #if pseudos[x][:6] == 'BTCUSD':
        #    print(ele)
        #pred_bull_wide   = apply_model(model_bull_wide, ele)[0]
        #pred_bear_wide   = apply_model(model_bear_wide, ele)[0]
        pred_bull_narrow = apply_model(model_bull_narrow, ele)[0]
        pred_bear_narrow = apply_model(model_bear_narrow, ele)[0]
        pred_bull_short  = apply_model(model_bull_short, ele)[0]
        pred_bear_short  = apply_model(model_bear_short, ele)[0]
        pred_bull_inter  = apply_model(model_bull_inter, ele)[0]
        pred_bear_inter  = apply_model(model_bear_inter, ele)[0]
        pred_bull_bulk   = apply_model(model_bull_bulk, ele)[0]
        pred_bear_bulk   = apply_model(model_bear_bulk, ele)[0]

        #pred_wide = get_pred(pred_bull_wide, pred_bear_wide, 'wide', 'threshold')
        pred_narrow = get_pred(pred_bull_narrow, pred_bear_narrow, 'narrow', 'threshold')
        pred_short = get_pred(pred_bull_short, pred_bear_short, 'short', 'threshold')
        pred_inter = get_pred(pred_bull_inter, pred_bear_inter, 'inter', 'threshold')
        pred_bulk = get_pred(pred_bull_bulk, pred_bear_bulk, 'bulk', 'threshold')

        #pred_wide_ = get_pred(pred_bull_wide, pred_bear_wide, 'wide', 'comb')
        pred_narrow_ = get_pred(pred_bull_narrow, pred_bear_narrow, 'narrow', 'comb')
        pred_short_ = get_pred(pred_bull_short, pred_bear_short, 'short', 'comb')
        pred_inter_ = get_pred(pred_bull_inter, pred_bear_inter, 'inter', 'comb')
        pred_bulk_ = get_pred(pred_bull_bulk, pred_bear_bulk, 'bulk', 'comb')

        pred = None
        if      pred_short == 2 and (pred_narrow == 2 or pred_bulk == 2 or pred_inter == 2):
            pred = 2
        elif    pred_short == 0  and (pred_narrow < 2 and pred_bulk < 2 and pred_inter < 2):
            pred = 0
        else:
            pred = 1

        pred_ = None
        if      pred_narrow_ == 2 and pred_short_ == 2 and pred_inter_ == 2 and pred_bulk_ == 2:
            pred_ = 2
        elif    pred_short_ == 0 and pred_bulk_ == 0 and  pred_narrow_ < 2 and pred_inter_ < 2:
            pred_ = 0
        else:
            pred_ = 1

        print(f"{pseudos[x][:6]} {str(pred_bulk)+str(pred_narrow)+str(pred_inter)+str(pred_short)+str(pred_)} : {scores[pseudos[x]][-1]}")


        if pseudos[x] not in last_orders:
            last_orders[pseudos[x]] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if check_pending_orders(pseudos[x]):
            if datetime.now() - datetime.strptime(last_orders[pseudos[x]], "%Y-%m-%d %H:%M:%S") > timedelta(minutes=unfilled_order_lifespan_min):
                cancel_unfilled_orders(pseudos[x])
            else:
                continue

        if pred == 2 or pred_ == 2:
            if check_open_positions(pseudos[x]):
                if datetime.now() - datetime.strptime(last_orders[pseudos[x]], "%Y-%m-%d %H:%M:%S") < timedelta(hours=hours_before_repeat_order):
                    continue
            if correlated(pseudos[x], ele):
                continue
            so(
                symbol      =   pseudos[x],
                ordertype   =   order_types_[0],
                volume      =   None,
                price       =   None,
                mode        =   dashboard['defaultTradingMode'],
                delta_timeframe_pair = delta_timeframe_pair_pseudos['h']
            )
            last_orders[pseudos[x]] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        elif pred == 0 or pred_ == 0:
            if check_open_positions(pseudos[x]):
                if datetime.now() - datetime.strptime(last_orders[pseudos[x]], "%Y-%m-%d %H:%M:%S") < timedelta(hours=hours_before_repeat_order):
                    continue
            if correlated(pseudos[x], ele):
                continue
            so(
                symbol      =   pseudos[x],
                ordertype   =   order_types_[1],
                volume      =   None,
                price       =   None,
                mode        =   dashboard['defaultTradingMode'],
                delta_timeframe_pair = delta_timeframe_pair_pseudos['h']
            )
            last_orders[pseudos[x]] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            pass
    iteration_time = datetime.now()
    sleep(60)
    print(f"======================================================================")

def main():
    init_models()
    init_env()
    while (True):
        loop()

if __name__ == '__main__':
    main()
