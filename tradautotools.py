import random
import tradparams as tparams
import os
from catboost import CatBoostClassifier


def get_model(model_name="activation008.cbm"):
    # Création d'un nouvel objet de modèle
    loaded_model = CatBoostClassifier()

    # Chargement du modèle depuis le fichier
    loaded_model.load_model(model_name)

    return loaded_model




class RiskTooHighException(Exception):
    """Exception raised when calculated risk exceeds acceptable limits."""
    def __init__(self, risk_value, message="Calculated minimum risk is too high for this order"):
        self.risk_value = risk_value
        self.message = f"{message}. Risk value: {risk_value} %"
        super().__init__(self.message)

class UnknownOrderTypeException(Exception):
    """Exception raised when the given order type is unknown."""
    def __init__(self, order_type, message="The given ordertype is unknown"):
        self.order_type = order_type
        self.message = f"{message}. Order_type: {order_type}"
        super().__init__(self.message)

class UnknownModeException(Exception):
    """Exception raised when the given order type is unknown."""
    def __init__(self, mode, message="The given mode is unknown"):
        self.order_type = mode
        self.message = f"{message}. Mode: {mode}"
        super().__init__(self.message)

def rmfile(filepath):
    # Supprimez le fichier
    if os.path.exists(filepath):
        os.remove(filepath)
        print("Fichier supprimé avec succès.")
    else:
        print("Le fichier n'existe pas.")

# Initialiser la connexion avec MetaTrader 5
def init_metatrader_connexion():
    if not tparams.mt5.initialize():
        print("Échec de l'initialisation de MetaTrader5")
        quit()

def check_symbol(symbol):
    if not tparams.mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")

def close_metatrader_connexion():
    tparams.mt5.shutdown()


def send_order(
    symbol, 
    order_type, 
    volume, 
    price=None, 
    stoploss=None, 
    takeprofit=None,
    typefilling=tparams.mt5.ORDER_FILLING_FOK
):
    """
    Envoie un ordre au marché via MetaTrader 5.

    Arguments:
    symbol -- le symbole du marché (ex: "EURUSD")
    order_type -- le type d'ordre ('buy', 'sell', 'buy_limit', 'sell_limit', 'buy_stop', 'sell_stop')
    volume -- la quantité d'actifs à acheter ou vendre
    price -- le prix pour les ordres limit et stop (non requis pour buy et sell)
    sl -- stop loss (optionnel)
    tp -- take profit (optionnel)
    """
    
    if order_type not in tparams.order_types:
        print("Type d'ordre non valide")
        return False

    init_metatrader_connexion()
    # Préparation de la requête
    request = {
        "action": tparams.mt5.TRADE_ACTION_DEAL if order_type in ['buy', 'sell'] else tparams.mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": tparams.order_types[order_type],
        "price": tparams.mt5.symbol_info_tick(symbol).ask if price is None else price,
        "sl": stoploss,
        "tp": takeprofit,
        "deviation": 10,  # Tolérance en points pour le slippage
        "magic": random.randint(1, 100000000000000),  # Identifiant unique de l'ordre
        "comment": f"{order_type} order",
        "type_time": tparams.mt5.ORDER_TIME_GTC,  # 'Good till cancelled'
        "type_filling": typefilling # tparams.mt5.ORDER_FILLING_FOK,  # 'Fill or kill'
    }
    
    check_symbol(symbol)
    result = tparams.mt5.order_send(request)
    close_metatrader_connexion()
    print(f"Sent order: {request}")
    if result.retcode != tparams.mt5.TRADE_RETCODE_DONE:
        print(f"Échec de l'envoi de l'ordre : {result.retcode}")
        return False
    print(f"Ordre {order_type} exécuté avec succès!")
    return True

def get_prices(symbol):
    init_metatrader_connexion()
    check_symbol(symbol)
    # Get symbol information
    symbol_info = tparams.mt5.symbol_info(symbol)

    # Check if the symbol is available for trading
    if symbol_info is None:
        print(f"{symbol} not found.")
        tparams.mt5.shutdown()
    else:
        # Ensure the symbol is active in the Market Watch
        if not symbol_info.visible:
            tparams.mt5.symbol_select(symbol, True)

        # Get the latest tick data for the symbol
        tick = tparams.mt5.symbol_info_tick(symbol)

        # Retrieve and print the bid and ask prices
        if tick:
            return tick.bid,tick.ask
        else:
            print("Failed to retrieve tick data.")
    close_metatrader_connexion()

def get_equity():
    init_metatrader_connexion()

    # Retrieve account information
    account_info = tparams.mt5.account_info()
    if account_info is not None:
        # Access the equity attribute
        equity = account_info.equity * tparams.dashboard['equity_limit_ratio'] / 100.0
    else:
        print("Failed to retrieve account information")

    # Shut down the connection to MetaTrader 5
    tparams.mt5.shutdown()
    return equity


from datetime import datetime, timedelta

def candle_size(symbol, delta_timeframe_pair):
    # Initialize MetaTrader5
    init_metatrader_connexion()
    
    # Define the time range to get candles for the last 24 hours
    now = datetime.now()
    t = now - timedelta(hours=delta_timeframe_pair[0])

    # Retrieve hourly candles for the symbol in this time range
    candles = tparams.mt5.copy_rates_range(symbol, delta_timeframe_pair[1], t, now)
    
    # Check if data was successfully retrieved
    if candles is None or len(candles) == 0:
        print(f"No candle data found for {symbol}")
        close_metatrader_connexion()
        return None

    # Calculate the average candle size
    candle_sizes = [candle['high'] - candle['low'] for candle in candles]
    total_size = sum(candle_sizes)
    average_size = total_size / len(candles)
    max_size = max(candle_sizes)

    close_metatrader_connexion()

    return max_size, average_size

def get_conv_pair(symbol, base=tparams.dashboard['base_currency']):
    pair = None
    if base=='EUR':
        pair = tparams.eur_conv_pairs[symbol[3:]]
    else:
        print(f"Symbol {symbol} doesn't have a conversion pair")
    return pair

def get_conversion_factor(symbol, base=tparams.dashboard['base_currency']):
    return get_prices(get_conv_pair(symbol, base))[0]


def get_minimal_lot_size(symbol):
    # Initialize MetaTrader5
    init_metatrader_connexion()
    
    # Get symbol information
    symbol_info = tparams.mt5.symbol_info(symbol)
    
    # Check if symbol information is retrieved successfully
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        tparams.mt5.shutdown()
        return None
    
    # Retrieve the lot size (contract size) for the symbol
    lot_size = symbol_info.volume_min

    close_metatrader_connexion()

    return lot_size

def get_volume_step(symbol):
    init_metatrader_connexion()

    # Symbol info
    info_symbole = tparams.mt5.symbol_info(symbol)
    
    if info_symbole is None:
        print(f"Error : symbol info unfound ({symbol})")
    else:
        volume_step = info_symbole.volume_step
    close_metatrader_connexion()
    return volume_step


def get_lot_value_and_currency(symbol):
    # Initialize MetaTrader5
    if not tparams.mt5.initialize():
        print("MetaTrader5 initialization failed")
        return None
    
    # Get symbol information
    symbol_info = tparams.mt5.symbol_info(symbol)
    
    # Check if symbol information is retrieved successfully
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        tparams.mt5.shutdown()
        return None
    
    # Get the lot value and the currency for the symbol
    lot_value = symbol_info.trade_contract_size
    currency = symbol_info.currency_profit

    # Close MetaTrader5 connection
    tparams.mt5.shutdown()

    return lot_value, currency


def get_contract_size(symbol):
    return get_lot_value_and_currency(symbol)[0]


def get_loss(risk, equity=get_equity()):
    loss = equity * risk / 100.0
    return loss

def get_risk_value_and_lot_size(symbol, loss_variance, contract_size, conversion_price, equity=get_equity(), risk_level=tparams.dashboard['risk_level'],accepted_risk_overrun=tparams.dashboard['accepted_risk_overrun']):
    desired_lot_size = get_minimal_lot_size(symbol)
    volume_step = get_volume_step(symbol)
    print(f"Minimum lot size = {desired_lot_size}")
    risk = None
    while True:
        test_risk = 100.0 * contract_size * loss_variance * desired_lot_size / equity / conversion_price
        if test_risk > risk_level:
            break
        else:
            risk = test_risk
            desired_lot_size = desired_lot_size + volume_step
    if risk == None:
        if test_risk - risk_level <= accepted_risk_overrun:
            risk = test_risk
        else:
            raise RiskTooHighException(test_risk)
    else:
        desired_lot_size = desired_lot_size - volume_step
    print(f"Max loss = {get_loss(risk)}")
    return risk, desired_lot_size


def get_attributes(
    symbol     ,
    ordertype  ,
    volume     ,
    price      ,
    mode       ,
    delta_timeframe_pair
):
    max_candle_size, avg_candle_size = candle_size(symbol, delta_timeframe_pair)

    if mode=='swing':
        loss_variance = max_candle_size   
    elif mode=='intraday':
        loss_variance = avg_candle_size
    else:
        raise UnknownModeException(mode)

    print(f"loss_variance = {loss_variance}")
          
    win_variance = tparams.dashboard['win_loss_quotient'] * loss_variance

    if ordertype in ['buy_now','sell_now']:
        offset = avg_candle_size * tparams.dashboard['min_offset_ratio']
    elif ordertype in ['buy','sell']:
        offset = avg_candle_size * tparams.dashboard['medium_low_offset_ratio']
    elif ordertype in ['buy_limit','sell_limit']:
        offset = avg_candle_size * tparams.dashboard['medium_high_offset_ratio']
    elif ordertype in ['buy_wide','sell_wide']:
        offset = avg_candle_size * tparams.dashboard['max_offset_ratio']
    elif ordertype in ['buy_stop','sell_stop']:
        offset = avg_candle_size * tparams.dashboard['stop_offset_ratio']
    else:
        offset = 0

    print(f"offset = {offset}")

    bid, ask = get_prices(symbol)
    order_type = None
    if ordertype in tparams.buy_orders:
        market_price = bid
        price = market_price - offset
        loss_price = price - loss_variance
        win_price = price + win_variance
        order_type = 'buy_limit'
        typefilling = tparams.mt5.ORDER_FILLING_FOK
    elif ordertype in tparams.sell_orders:
        market_price = ask
        price = market_price + offset
        loss_price = price + loss_variance
        win_price = price - win_variance
        order_type = 'sell_limit'
        typefilling = tparams.mt5.ORDER_FILLING_FOK


    elif ordertype == 'buy_stop':
        market_price = ask
        price = market_price + offset
        loss_price = price - loss_variance
        win_price = price + win_variance
        order_type = 'buy_stop'
        typefilling = tparams.mt5.ORDER_FILLING_FOK
    elif ordertype == 'sell_stop':
        market_price = bid
        price = market_price - offset
        loss_price = price + loss_variance
        win_price = price - win_variance
        order_type = 'sell_stop'
        typefilling = tparams.mt5.ORDER_FILLING_FOK


    elif ordertype == 'buy_market':
        loss_price = ask - loss_variance
        win_price = 2 * ask + win_variance - bid
        order_type = 'buy'
        price = None
        typefilling = tparams.mt5.ORDER_FILLING_IOC
    elif ordertype == 'sell_market':
        loss_price = bid + loss_variance
        win_price = 2 * bid - win_variance - ask
        order_type = 'sell'
        price = None
        typefilling = tparams.mt5.ORDER_FILLING_IOC      
    else:
        raise UnknownOrderTypeException(ordertype)


    contract_size = get_contract_size(symbol)

    if symbol[:3] == tparams.dashboard['base_currency']:
        conversion_price = loss_price
    else:
        conversion_price = 1 # get_conversion_factor(symbol)
    print(f"Conversion price = {conversion_price}")

    if not volume:
        risk, calc_volume = get_risk_value_and_lot_size(symbol, loss_variance, contract_size, conversion_price)

    print(f"risk = {risk} %")

    return order_type, calc_volume, price, loss_price, win_price, typefilling