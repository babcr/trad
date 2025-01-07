#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from collections import deque
from tradcreatemldata import minute_in_year_normalization
from datetime import datetime, timedelta
from time import sleep
from numpy import void, float64
from so import main as so
from tradautotools import init_metatrader_connexion, close_metatrader_connexion
from tradparams import order_types_, period, mperiod, pseudos, pseudos_we, mt5, ranges_equi, prediction_period, mean_period, learning_rate, percentile, modelfile_extension
from tradparams import limit_correlation, dashboard, delta_timeframe_pair_pseudos
from tradparams import unfilled_order_lifespan_min, hours_before_repeat_order
from tradparams import percs, special_percs, used_timeframes, ranges, directions, modes, initial_preds
from tradparams import xgb
import copy
from numpy import ndarray,corrcoef

maxtry = 10
present_times = {}
treatments_returns = {}
last_orders = {}
start_of_friday = None
end_of_sunday = None
iteration_time = datetime.now()
first = True

def calculate_candlestick_score_realtime(symbol, last_candle, base_sum_supports, base_sums, meanperiod=mperiod, new=True):
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

def load_candles(symbol, timeframe, back_period=mperiod):
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
        cands = mt5.copy_rates_from_pos(symbol, timeframe, 0, back_period)  # Exemple TIMEFRAME_D1 pour des chandeliers quotidiens

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


def load_candle(symbol, candles, scores, base_sum_supports, base_sums, timeframe):
    # Initialiser MetaTrader5
    init_metatrader_connexion()

    ok = False
    k = 0
    candle = None
    while not ok:
        # Télécharger les chandeliers
        candle = mt5.copy_rates_from_pos(symbol, timeframe, 0, 1)
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
        candles[symbol] = deque([candle])
        scores[symbol] = deque([calculate_candlestick_score_realtime(symbol, candle, base_sum_supports, base_sums)])
        present_times[symbol] = t
    elif current_candle_time > last_candle_time:
        #print(f"Nouvelle bougie détectée : {datetime.fromtimestamp(current_candle_time)}")
        if type(candles[symbol]) == ndarray:
            candles[symbol] = deque(candles[symbol])
        candles[symbol].append(candle)
        scores[symbol].append(calculate_candlestick_score_realtime(symbol, candle, base_sum_supports, base_sums))
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
        new = calculate_candlestick_score_realtime(symbol, candle, base_sum_supports, base_sums, new=False)
        scores[symbol].append(new)
        present_times[symbol] = t

    close_metatrader_connexion()
    return scores[symbol]


def init_env(timeframe, candles, scores, base_sum_supports, base_sums):
    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for x in pseudos:
        candles[pseudos[x]] = load_candles(pseudos[x], timeframe=timeframe)
        scores[pseudos[x]] = deque()
        for i in range(0, period):
            scores[pseudos[x]].append(
                calculate_candlestick_score_realtime(pseudos[x], candles[pseudos[x]][mperiod - period + i], base_sum_supports, base_sums)
            )
            present_times[pseudos[x]] = t

def load_element(symbol, scores):
    myn = minute_in_year_normalization(present_times[symbol])
    ele = scores[symbol].copy()
    ele.appendleft(myn)
    #print(f"ele size = {len(ele)}")
    return [list(ele)]

def apply_model(model, element):
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

def get_pred(bull_pred, bear_pred, scale, keyword, prob_limits, rev=False):
    if   bull_pred > prob_limits[f'bull_binary_{scale}_{'rev_' * rev}{keyword}'] and bear_pred <= prob_limits[f'bear_binary_{scale}_{keyword}']:
        pred = 2
    elif bear_pred > prob_limits[f'bear_binary_{scale}_{keyword}'] and bull_pred <= prob_limits[f'bull_binary_{scale}_{'rev_' *  rev}{keyword}']:
        pred = 0
    else:
        pred = 1
    return pred

def get_pred_rev(bull_pred, bear_pred, scale, keyword):
    if   bull_pred > dashboard[f'bull_binary_{scale}_rev_{keyword}'] and bear_pred <= dashboard[f'bear_binary_{scale}_{keyword}']:
        pred = 2
    elif bear_pred > dashboard[f'bear_binary_{scale}_{keyword}'] and bull_pred <= dashboard[f'bull_binary_{scale}_rev_{keyword}']:
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

def correlated(symbol, ele, scores):
    for op in get_active_symbols():
        if op != symbol:
            if corr(ele, load_element(op, scores)) > limit_correlation:
                print(f"correlation {op} and {symbol} ==> {corr(ele, load_element(op, scores))}")
                return True
    return False

def is_in_last_seconds_of_x_minutes(x):
    now = datetime.now()
    y = 60 * now.hour + now.minute
    return y % x == x - 1 or y % x == x - 2

def set_primary_preds(symbol : str, preds : dict, models : dict, ele : list, direction : str, rang : str, rev : bool):
    for x in directions:
        if x != direction:
            codirection = x
    preds[symbol][f"pred_{'rev_' * rev}{direction}_{rang}"] = apply_model(models[f'model_{direction * (not rev)}{codirection * rev}_{rang}'], ele)[0]

def set_secondary_preds(symbol : str, preds : dict, mode : str, rang : str, prob_limits : dict, rev : bool):
    preds[symbol][f"pred_{'rev_' * rev}{rang}{(mode=='comb') * '_'}"] = get_pred(preds[symbol][f"pred_{'rev_' * rev}{'bull' * (not rev)}{'bear' * rev}_{rang}"], preds[symbol][f"pred_{'rev_' * rev}{'bull' * rev}{'bear' * (not rev)}_{rang}"] , rang, mode, prob_limits, rev)


def set_preds(symbol : str, preds : dict, models : dict, ele : list,  prob_limits : dict, scores : dict):
    opele = [[-x for x in ele[0]]]
    for direction in directions:
        for rang in ranges:
            set_primary_preds(symbol, preds, models, ele, direction, rang, False)
            set_primary_preds(symbol, preds, models, opele, direction, rang, True)

    for mode in modes:
        for rang in ranges:
            set_secondary_preds(symbol, preds, mode, rang, prob_limits, False)
            set_secondary_preds(symbol, preds, mode, rang, prob_limits, True)


    preds[symbol]["pred"] = None
    if      (preds[symbol]["pred_short"] == 2 or preds[symbol]["pred_inter"] == 2) and (preds[symbol]["pred_narrow"] == 2 or preds[symbol]["pred_bulk"] == 2 ):
        preds[symbol]["pred"] = 2
    elif    preds[symbol]["pred_short"] == 0  and (preds[symbol]["pred_narrow"] < 2 and preds[symbol]["pred_bulk"] < 2 and preds[symbol]["pred_inter"] < 2):
        preds[symbol]["pred"] = 0
    else:
        preds[symbol]["pred"] = 1

    preds[symbol]["pred_"] = None
    if      preds[symbol]["pred_narrow_"] == 2 and preds[symbol]["pred_short_"] == 2 and preds[symbol]["pred_inter_"] == 2 and preds[symbol]["pred_bulk_"] == 2:
        preds[symbol]["pred_"] = 2
    elif    preds[symbol]["pred_short_"] == 0 and preds[symbol]["pred_bulk_"] == 0 and preds[symbol]["pred_narrow"] < 2 and preds[symbol]["pred_inter"] < 2:
        preds[symbol]["pred_"] = 0
    else:
        preds[symbol]["pred_"] = 1

    preds[symbol]["pred_rev"] = None
    if      (preds[symbol]["pred_rev_short"] == 2 or preds[symbol]["pred_rev_inter"] == 2) and (preds[symbol]["pred_rev_narrow"] == 2 or preds[symbol]["pred_rev_bulk"] == 2 ):
        preds[symbol]["pred_rev"] = 0
    elif    preds[symbol]["pred_rev_short"] == 0  and (preds[symbol]["pred_rev_narrow"] < 2 and preds[symbol]["pred_rev_bulk"] < 2 and preds[symbol]["pred_rev_inter"] < 2):
        preds[symbol]["pred_rev"] = 1
    else:
        preds[symbol]["pred_rev"] = 1

    preds[symbol]["pred_rev_"] = None
    if      preds[symbol]["pred_rev_narrow_"] == 2 and preds[symbol]["pred_rev_short_"] == 2 and preds[symbol]["pred_rev_inter_"] == 2 and preds[symbol]["pred_rev_bulk_"] == 2:
        preds[symbol]["pred_rev_"] = 0
    elif    preds[symbol]["pred_rev_short_"] == 0 and preds[symbol]["pred_rev_bulk_"] == 0 and preds[symbol]["pred_rev_narrow"] < 2 and preds[symbol]["pred_rev_inter"] < 2:
        preds[symbol]["pred_rev_"] = 2
    else:
        preds[symbol]["pred_rev_"] = 1


    return f"{symbol[:6]} {str(preds[symbol]["pred"] )} {str(preds[symbol]["pred_bulk"] )+str(preds[symbol]["pred_narrow"] )+str(preds[symbol]["pred_inter"] )+str(preds[symbol]["pred_short"] )} - {str(preds[symbol]["pred_"] )} {str(preds[symbol]["pred_bulk_"] )+str(preds[symbol]["pred_narrow_"] )+str(preds[symbol]["pred_inter_"] )+str(preds[symbol]["pred_short_"] )} | {str(preds[symbol]["pred_rev"])} {str(preds[symbol]["pred_rev_bulk"] )+str(preds[symbol]["pred_rev_narrow"] )+str(preds[symbol]["pred_rev_inter"] )+str(preds[symbol]["pred_rev_short"] )} - {str(preds[symbol]["pred_rev_"] )} {str(preds[symbol]["pred_rev_bulk_"] )+str(preds[symbol]["pred_rev_narrow_"] )+str(preds[symbol]["pred_rev_inter_"] )+str(preds[symbol]["pred_rev_short_"] )}: {scores[symbol][-1]}"

def execute_order(preds, symbol, ele, scores, timeframe_pseudo='h'):
    global last_orders
    if symbol not in last_orders:
        last_orders[symbol] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if check_pending_orders(symbol):
        if datetime.now() - datetime.strptime(last_orders[symbol], "%Y-%m-%d %H:%M:%S") > timedelta(minutes=unfilled_order_lifespan_min):
            cancel_unfilled_orders(symbol)
        else:
            return -1

    if ((preds[symbol]["pred"] == 2 or preds[symbol]["pred_"] == 2) and (preds[symbol]["pred_rev"] > 0 and preds[symbol]["pred_rev_"] > 0)) or ((preds[symbol]["pred_rev"] == 2 or preds[symbol]["pred_rev_"] == 2) and (preds[symbol]["pred"] > 0 and preds[symbol]["pred_"] > 0)):
        if check_open_positions(symbol):
            if datetime.now() - datetime.strptime(last_orders[symbol], "%Y-%m-%d %H:%M:%S") < timedelta(hours=hours_before_repeat_order):
                return -1
        if correlated(symbol, ele, scores):
            return -1
        so(
            symbol      =   symbol,
            ordertype   =   order_types_[0],
            volume      =   None,
            price       =   None,
            delta_timeframe_pair = delta_timeframe_pair_pseudos[timeframe_pseudo]
        )
        last_orders[symbol] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elif (
        (preds[symbol]["pred"] == 0 or preds[symbol]["pred_"] == 0) 
        and (preds[symbol]["pred_rev"] < 2 and preds[symbol]["pred_rev_"] < 2)
        ) or (
            (preds[symbol]["pred_rev"] == 0 or preds[symbol]["pred_rev_"] == 0) and 
            (preds[symbol]["pred"] < 2 and preds[symbol]["pred_"] < 2)
        ):
        if check_open_positions(symbol):
            if datetime.now() - datetime.strptime(last_orders[symbol], "%Y-%m-%d %H:%M:%S") < timedelta(hours=hours_before_repeat_order):
                return -1
        if correlated(symbol, ele, scores):
            return -1
        so(
            symbol      =   symbol,
            ordertype   =   order_types_[1],
            volume      =   None,
            price       =   None,
            delta_timeframe_pair = delta_timeframe_pair_pseudos[timeframe_pseudo]
        )
        last_orders[symbol] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return 0


def init_op_dict():
    global percs, special_percs
    opd = {}
    for timeframe_pseudo in used_timeframes:
        i_opd = {}
        for p in percs:
            if p in special_percs[timeframe_pseudo]:
                i_opd[p] = special_percs[timeframe_pseudo][p]
            else:
                i_opd[p] = {}
        opd[timeframe_pseudo] = i_opd
    return opd

def load_model(model_name):
    # Load the model from the file
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_name)
    return loaded_model

operating_dicts = {}
def init_models(
        timeframe_pseudo,
        prediction_period, 
        mean_period,
        learning_rate, 
        percentile,
        modelfile_extension
    ):
    global operating_dicts
    models = {}

    if timeframe_pseudo not in models:
        models[timeframe_pseudo] = {}
    for r in ranges:
        for d in directions:
            key = f'model_{d}_{r}'
            model_in_use = f"M{prediction_period}_{mean_period}_{learning_rate}_{percentile}_{d}_{r}_{ranges_equi[r]}{f'_{timeframe_pseudo}'*(timeframe_pseudo != 'h')}{modelfile_extension}"
            print(model_in_use)
            models[timeframe_pseudo][key] = load_model(model_in_use)

    operating_dicts[timeframe_pseudo]['models'] = models[timeframe_pseudo]


def treatment(x, pse, timeframe_pseudo):
    global first, treatments_returns, iteration_time, operating_dicts
    if datetime.now() - iteration_time > timedelta(minutes=delta_timeframe_pair_pseudos[timeframe_pseudo][2]):
        init_env(
            delta_timeframe_pair_pseudos[timeframe_pseudo][1],
            operating_dicts[timeframe_pseudo]['candles'],
            operating_dicts[timeframe_pseudo]['scores'],
            operating_dicts[timeframe_pseudo]['base_sum_supports'],
            operating_dicts[timeframe_pseudo]['base_sums']
        )
    ele = None
    if is_in_last_seconds_of_x_minutes(delta_timeframe_pair_pseudos[timeframe_pseudo][2]) or first == True:
        if x == next(iter(pse.items()))[0]:
            print("================================================================")
        if pse[x] not in operating_dicts[timeframe_pseudo]['preds']:
            operating_dicts[timeframe_pseudo]['preds'][pse[x]] = copy.deepcopy(initial_preds)
        load_candle(pse[x], operating_dicts[timeframe_pseudo]['candles'], operating_dicts[timeframe_pseudo]['scores'], operating_dicts[timeframe_pseudo]['base_sum_supports'], operating_dicts[timeframe_pseudo]['base_sums'], timeframe=delta_timeframe_pair_pseudos[timeframe_pseudo][1])
        ele = load_element(pse[x], operating_dicts[timeframe_pseudo]['scores'])
        if pse[x] not in treatments_returns:
            treatments_returns[pse[x]] = {}
        if timeframe_pseudo not in treatments_returns[pse[x]]:
            treatments_returns[pse[x]][timeframe_pseudo] = {}
        treatments_returns[pse[x]][timeframe_pseudo]['res'] = set_preds(pse[x], operating_dicts[timeframe_pseudo]['preds'], operating_dicts[timeframe_pseudo]['models'], ele, operating_dicts[timeframe_pseudo]['prob_limits'], operating_dicts[timeframe_pseudo]['scores'])    
        print(f"{treatments_returns[pse[x]][timeframe_pseudo]['res']}")
    
    if ele:
        treatments_returns[pse[x]][timeframe_pseudo]['r'] = execute_order(operating_dicts[timeframe_pseudo]['preds'], pse[x], ele, operating_dicts[timeframe_pseudo]['scores'], timeframe_pseudo)


def loop():
    global iteration_time, operating_dicts
    global first

    pse = None
    if is_weekend():
        pse = pseudos_we
    else:
        pse = pseudos

    for x in pse:
        for timeframe_pseudo in used_timeframes:
            treatment(x, pse, timeframe_pseudo)

    iteration_time = datetime.now()
    first = False
    sleep(30)

def main():
    global operating_dicts
    operating_dicts = init_op_dict()
    for timeframe_pseudo in used_timeframes:
        init_models(
            timeframe_pseudo,
            prediction_period, 
            mean_period,
            learning_rate,
            percentile,
            modelfile_extension  
        )
        init_env(
            delta_timeframe_pair_pseudos[timeframe_pseudo][1],
            operating_dicts[timeframe_pseudo]['candles'],
            operating_dicts[timeframe_pseudo]['scores'],
            operating_dicts[timeframe_pseudo]['base_sum_supports'],
            operating_dicts[timeframe_pseudo]['base_sums']
        )
    while (True):
        loop()

if __name__ == '__main__':
    main()
