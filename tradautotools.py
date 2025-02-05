from random import randint
from tradparams import granularity_factor, order_suffix, deviation, max_spread, limit_spread, mt5, order_types, dashboard, eur_conv_pairs, buy_orders, sell_orders
from tradparams import delta_timeframe_pair_pseudos, orders_list, symbols_list, pseudos, symbol_converter
from math import ceil, floor
from os import path, remove, listdir, path
from datetime import datetime, timedelta

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

class SpreadTooHighException(Exception):
    """Exception raised when the given order type is unknown."""
    def __init__(self, spread, message=f"The calculated spread is higher than {max_spread}"):
        self.spread = spread
        self.message = f"{message}. Spread: {self.spread}"
        super().__init__(self.message)

class MaxPositionsException(Exception):
    """Exception raised when the maximum number of positions is reached."""
    def __init__(self, positions, message=f"The maximum number of positions is reached ({floor(100.0 / dashboard["risk_level"])})"):
        self.positions = positions
        self.message = f"{message}. Orders: {self.positions}"
        super().__init__(self.message)

class PendingOrderInlineException(Exception):
    """Exception raised when the maximum number of positions is reached."""
    def __init__(self, symbol, message=f"An order already exists for that symbol."):
        self.symbol = symbol
        self.message = f"{message}. Symbol: {self.symbol}"
        super().__init__(self.message)

class UnicityError(Exception):
    """Exception raised when 2 files with the same nomination are found."""
    def __init__(self, starting_word, matching_files, message="The given starting word is held by two distinct files in the folder"):
        self.starting_word = starting_word
        self.matching_files = matching_files
        self.message = f"{message}. starting_word: {starting_word}, matching_files: {matching_files}"
        super().__init__(self.message)

def find_file_by_starting_words(directory, starting_words):
    """
    Finds a file in a directory that starts with the given words.

    :param directory: The directory to search in
    :param starting_words: The starting words of the file name
    :return: The file name if uniquely found
    :raises UnicityError: If more than one file matches the starting words
    :raises FileNotFoundError: If no file matches the starting words
    """
    if not path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    
    if not path.isdir(directory):
        raise NotADirectoryError(f"The path '{directory}' is not a directory.")

    # Find all files that start with the given starting words
    matching_files = [
        file for file in listdir(directory) if file.startswith(starting_words)
    ]

    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file starts with '{starting_words}'.")
    elif len(matching_files) > 1:
        raise UnicityError(starting_words, matching_files)
    
    return matching_files[0]

def extract_last_number(filename):
    """
    Extracts the last word or number in the file name that contains at least one digit.
    
    :param filename: The file name with words and numbers separated by '_'
    :return: The last word or number containing at least one digit, or None if none is found
    """
    # Split the filename into parts by '_'
    parts = filename.split('_')
    
    # Loop through parts in reverse and find the first that contains at least one digit
    for part in reversed(parts):
        if all(char.isdigit() for char in part.split('.')[0]):
            return part.split('.')[0]
    
    return None  # Return None if no valid part is found

def delete_files_in_directory(path):
    """
    Deletes all files in the specified directory.

    :param path: Path to the directory
    """
    if not path.exists(path):
        print(f"The specified path does not exist: {path}")
        return

    if not path.isdir(path):
        print(f"The specified path is not a directory: {path}")
        return

    try:
        for file in listdir(path):
            full_path = path.join(path, file)
            if path.isfile(full_path):
                remove(full_path)
                print(f"Deleted file: {full_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_suffix(directory, starting_words):
    filen = find_file_by_starting_words(directory, starting_words)
    suffix = extract_last_number(filen)
    return suffix, filen

def rmfile(filepath):
    # Supprimez le fichier
    if path.exists(filepath):
        remove(filepath)
        print(f"File {filepath} successfully deleted.")
    else:
        print(f"File {filepath} doesn't exist.")

# Initialiser la connexion avec MetaTrader 5
def init_metatrader_connexion():
    if not mt5.initialize():
        print("Failed MetaTrader5 initialisation")
        quit()

def check_symbol(symbol):
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")

def close_metatrader_connexion():
    mt5.shutdown()


def calculate_start_year(timeframe, x, points_per_vector, current_date=None):
    """
    Calcule l'année à partir de laquelle il faut charger des données Forex pour un symbol donné.
    
    :param symbol: (str) Le symbole Forex (par exemple "EURUSD").
    :param timeframe: (int) Le timeframe de MetaTrader 5 (ex. mt5.TIMEFRAME_M1).
    :param x: (int) Nombre de vecteurs souhaités.
    :param points_per_vector: (int) Nombre de valeurs par vecteur.
    :param current_date: (datetime) Date actuelle pour référence. Si None, prend la date actuelle.
    :return: (int) Année à partir de laquelle commencer le chargement des données.
    """
    # Utiliser la date actuelle si aucune n'est spécifiée
    if current_date is None:
        current_date = datetime.now()

    # Total de points nécessaires
    total_points = x + points_per_vector

    # Obtenir la durée d'un point pour le timeframe spécifié
    timeframe_durations = {
        mt5.TIMEFRAME_M1: timedelta(minutes=1),
        mt5.TIMEFRAME_M5: timedelta(minutes=5),
        mt5.TIMEFRAME_M15: timedelta(minutes=15),
        mt5.TIMEFRAME_M30: timedelta(minutes=30),
        mt5.TIMEFRAME_H1: timedelta(hours=1),
        mt5.TIMEFRAME_H2: timedelta(hours=2),
        mt5.TIMEFRAME_H3: timedelta(hours=3),
        mt5.TIMEFRAME_H4: timedelta(hours=4),
        mt5.TIMEFRAME_H6: timedelta(hours=6),
        mt5.TIMEFRAME_H8: timedelta(hours=8),
        mt5.TIMEFRAME_H12: timedelta(hours=12),
        mt5.TIMEFRAME_D1: timedelta(days=1)
    }
    if timeframe not in timeframe_durations:
        raise ValueError("Timeframe non pris en charge.")

    point_duration = timeframe_durations[timeframe]

    # Calculer la période totale nécessaire
    total_duration = point_duration * total_points * (1 + 2 / 7)

    # Calculer la date de début
    start_date = current_date - total_duration

    # Retourner l'année de la date de début
    return start_date.year



def send_order(
    symbol,
    order_type,
    volume,
    price=None,
    stoploss=None,
    takeprofit=None,
    typefilling=mt5.ORDER_FILLING_FOK
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

    if order_type not in order_types:
        print("Type d'ordre non valide")
        return False

    init_metatrader_connexion()
    # Préparation de la requête
    request = {
        "action": mt5.TRADE_ACTION_DEAL if order_type in ['buy', 'sell'] else mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": order_types[order_type],
        "price": mt5.symbol_info_tick(symbol).ask if price is None else price,
        "sl": stoploss,
        "tp": takeprofit,
        "deviation": deviation,  # Tolérance en points pour le slippage
        "type_time": mt5.ORDER_TIME_GTC,  # 'Good till cancelled'
        "type_filling": typefilling, # mt5.ORDER_FILLING_FOK,  # 'Fill or kill'
        "magic": randint(1, 10000000000),  # Identifiant unique de l'ordre
        "comment": f"{order_type} order"
    }

    check_symbol(symbol)
    result = mt5.order_send(request)
    close_metatrader_connexion()
    print(f"Sent order: {request}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Échec de l'envoi de l'ordre : {result.retcode}")
        return False
    print(f"Ordre {order_type} exécuté avec succès!")
    return True

def get_prices(symbol):
    init_metatrader_connexion()
    check_symbol(symbol)
    # Get symbol information
    symbol_info = mt5.symbol_info(symbol)

    # Check if the symbol is available for trading
    if symbol_info is None:
        print(f"{symbol} not found.")
        mt5.shutdown()
    else:
        # Ensure the symbol is active in the Market Watch
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        # Get the latest tick data for the symbol
        tick = mt5.symbol_info_tick(symbol)

        # Retrieve and print the bid and ask prices
        if tick:
            return tick.bid,tick.ask
        else:
            print("Failed to retrieve tick data.")
    close_metatrader_connexion()

def get_equity(equitylimit=dashboard['equity_limit_ratio']):
    init_metatrader_connexion()
    # Retrieve account information
    account_info = mt5.account_info()
    equity=None
    if account_info is not None:
        # Access the equity attribute
        equity = account_info.equity * equitylimit
    else:
        print("Failed to retrieve account information")
        close_metatrader_connexion()
        return None
    # Shut down the connection to MetaTrader 5
    close_metatrader_connexion()
    return equity

def candle_size(symbol, delta_timeframe_pair):
    # Initialize MetaTrader5
    init_metatrader_connexion()

    # Define the time range to get candles for the last 24 hours
    now = datetime.now()
    t = now - timedelta(hours=delta_timeframe_pair[0])

    # Retrieve hourly candles for the symbol in this time range
    candles = mt5.copy_rates_range(symbol, delta_timeframe_pair[1], t, now)

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


def get_conversion_rate(symbol, base=dashboard['base_currency']):
    # Connexion à MT5
    init_metatrader_connexion()
    
    # Récupérer les informations sur le symbole
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"Symbole {symbol} non trouvé")
        close_metatrader_connexion()
        return None
    
    if symbol_info.currency_profit != base:
        conversion_rate_symbol = f"{symbol_info.currency_profit}{base}"
        conversion_rate_info = mt5.symbol_info_tick(conversion_rate_symbol)
        if conversion_rate_info is None:
            conversion_rate_symbol = f"{base}{symbol_info.currency_profit}"
            conversion_rate_info = mt5.symbol_info_tick(conversion_rate_symbol)
            if conversion_rate_info is None:
                print(f"Taux de conversion non disponible pour {conversion_rate_symbol}")
                close_metatrader_connexion()
                return None
            else:
                conversion_rate = 1 / conversion_rate_info.bid
        else:
            conversion_rate = conversion_rate_info.bid
    else:
        conversion_rate = 1.0

    close_metatrader_connexion()
    return conversion_rate

#def get_conversion_factor(symbol, base=dashboard['base_currency']):
#    return get_prices(get_conv_pair(symbol, base))[0]


def get_minimal_lot_size(symbol):
    # Initialize MetaTrader5
    init_metatrader_connexion()

    # Get symbol information
    symbol_info = mt5.symbol_info(symbol)

    # Check if symbol information is retrieved successfully
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        close_metatrader_connexion()
        return None

    # Retrieve the lot size (contract size) for the symbol
    lot_size = symbol_info.volume_min

    close_metatrader_connexion()

    return lot_size

def get_volume_step(symbol):
    init_metatrader_connexion()

    # Symbol info
    info_symbole = mt5.symbol_info(symbol)

    if info_symbole is None:
        print(f"Error : symbol info unfound ({symbol})")
        close_metatrader_connexion()
        return None
    else:
        volume_step = info_symbole.volume_step
    close_metatrader_connexion()
    return volume_step


def get_lot_value_and_currency(symbol):
    # Initialize MetaTrader5
    init_metatrader_connexion()

    # Get symbol information
    symbol_info = mt5.symbol_info(symbol)

    # Check if symbol information is retrieved successfully
    if symbol_info is None:
        print(f"Symbol {symbol} not found")
        close_metatrader_connexion()
        return None

    # Get the lot value and the currency for the symbol
    lot_value = symbol_info.trade_contract_size
    currency = symbol_info.currency_profit

    # Close MetaTrader5 connection
    close_metatrader_connexion()

    return lot_value, currency


def get_contract_size(symbol):
    return get_lot_value_and_currency(symbol)[0]


def get_loss(risk, equity=get_equity()):
    loss = equity * risk / 100.0
    return loss


def get_risk_value_and_lot_size(symbol, loss_variance, contract_size, conversion_price, equity=get_equity(), risk_level=dashboard['risk_level'],accepted_risk_overrun=dashboard['accepted_risk_overrun']):
    desired_lot_size = get_minimal_lot_size(symbol)
    volume_step = get_volume_step(symbol)
    print(f"Minimum lot size = {desired_lot_size}")
    risk = None
    while True:
        test_risk = 100.0 * contract_size * loss_variance * desired_lot_size / equity * conversion_price
        if test_risk > risk_level:
            break
        else:
            risk = test_risk
            desired_lot_size += volume_step
    if risk == None:
        if test_risk - risk_level <= accepted_risk_overrun:
            risk = test_risk
        else:
            raise RiskTooHighException(test_risk)
    else:
        desired_lot_size -= volume_step
    print(f"Max loss = {get_loss(risk)}")
    return risk, desired_lot_size


def get_loss_var(symbol, delta_timeframe_pair, raw_spread, granularity_fact=granularity_factor, timescale=dashboard['defaultTradingMode'], loss_shrink=dashboard['loss_shrink_ratio'], offset_shrink=dashboard['offset_shrink_ratio'], loss_to_win=dashboard['win_loss_quotient']):
    max_candle_size, avg_candle_size = candle_size(symbol, delta_timeframe_pair)
    scale_factor = 1 #delta_timeframe_pair_pseudos['h'][0] / delta_timeframe_pair[0]

    if timescale=='swing':
        loss_var = scale_factor * ceil(granularity_fact * max_candle_size / avg_candle_size) * avg_candle_size * loss_shrink / granularity_fact
        offset =  scale_factor * ceil(granularity_fact * max_candle_size / avg_candle_size) * avg_candle_size * offset_shrink / granularity_fact
    elif timescale=='intraday':
        loss_var = scale_factor * floor(granularity_fact * max_candle_size / avg_candle_size) * avg_candle_size * loss_shrink / granularity_fact
        offset =  scale_factor * floor(granularity_fact * max_candle_size / avg_candle_size) * avg_candle_size * offset_shrink / granularity_fact
    else:
        raise UnknownModeException(timescale)
    return loss_var, loss_to_win * (loss_var + raw_spread) , offset

def get_attributes(
    symbol     ,
    ordertype  ,
    volume     ,
    price      ,
    delta_timeframe_pair
):
    bid, ask = get_prices(symbol)
    raw_spread = ask - bid
    loss_variance, win_variance, offset = get_loss_var(symbol, delta_timeframe_pair, raw_spread)

    print(f"loss_variance = {loss_variance}")

    print(f"offset = {offset}")

    
    spread = raw_spread / loss_variance
    if spread > limit_spread:
        ordertype = ordertype.replace('_market', order_suffix)

    if spread > max_spread:
        raise SpreadTooHighException(spread)

    order_type = None
    if ordertype in buy_orders:
        market_price = bid
        price = market_price - offset
        loss_price = price - loss_variance
        win_price = price + win_variance
        order_type = 'buy_limit'
        typefilling = mt5.ORDER_FILLING_FOK
    elif ordertype in sell_orders:
        market_price = ask
        price = market_price + offset
        loss_price = price + loss_variance
        win_price = price - win_variance
        order_type = 'sell_limit'
        typefilling = mt5.ORDER_FILLING_FOK


    elif ordertype == 'buy_stop':
        market_price = ask
        price = market_price + offset
        loss_price = price - loss_variance
        win_price = price + win_variance
        order_type = 'buy_stop'
        typefilling = mt5.ORDER_FILLING_FOK
    elif ordertype == 'sell_stop':
        market_price = bid
        price = market_price - offset
        loss_price = price + loss_variance
        win_price = price - win_variance
        order_type = 'sell_stop'
        typefilling = mt5.ORDER_FILLING_FOK


    elif ordertype == 'buy_market':
        loss_price = ask - loss_variance
        win_price = 2 * ask + win_variance - bid
        order_type = 'buy'
        price = None
        typefilling = mt5.ORDER_FILLING_IOC
    elif ordertype == 'sell_market':
        loss_price = bid + loss_variance
        win_price = 2 * bid - win_variance - ask
        order_type = 'sell'
        price = None
        typefilling = mt5.ORDER_FILLING_IOC
    else:
        raise UnknownOrderTypeException(ordertype)


    contract_size = get_contract_size(symbol)
    conversion_rate = get_conversion_rate(symbol)

    risk = None
    calc_volume = None
    if not volume:
        risk, calc_volume = get_risk_value_and_lot_size(symbol, loss_variance, contract_size, conversion_rate)

    print(f"risk = {risk} %")

    return order_type, calc_volume, price, loss_price, win_price, typefilling