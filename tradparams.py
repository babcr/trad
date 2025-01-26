import MetaTrader5 as mt5
from math import ceil, floor
from numpy import float16

def to_float(val):
    val = float16(val)
    return val

dashboard = {
    'win_loss_quotient'         : 2.25,
    'equity_limit_ratio'        : 100.0,
    'goal'                      : 300,
    'risk_level'                : 10.0,
    'accepted_risk_overrun'     : 0.50,
    'min_offset_ratio'          : 0.05,
    'medium_low_offset_ratio'   : 0.2,
    'medium_high_offset_ratio'  : 0.5,
    'max_offset_ratio'          : 1,
    'stop_offset_ratio'         : 2,
    'defaultTradingMode'        : 'intraday',
    'defaultDeltaTimeframePair' : 'h',
    'base_currency'             : 'EUR'
}

initial_thresh = 0.5

prediction_period = 30 # in days
minutes_in_a_year = 525600
min5    = (24 * prediction_period / 12  , mt5.TIMEFRAME_M5 , 1/12  *  60)
min15   = (24 * prediction_period / 4   , mt5.TIMEFRAME_M15, 0.25  *  60)
hourly  = (24 * prediction_period       , mt5.TIMEFRAME_H1 , 1     *  60)
hour2   = (24 * prediction_period * 2   , mt5.TIMEFRAME_H2 , 2     *  60)
hour3   = (24 * prediction_period * 3   , mt5.TIMEFRAME_H3 , 3     *  60)
hour4   = (24 * prediction_period * 4   , mt5.TIMEFRAME_H4 , 4     *  60)
hour6   = (24 * prediction_period * 6   , mt5.TIMEFRAME_H6 , 6     *  60)
hour8   = (24 * prediction_period * 8   , mt5.TIMEFRAME_H8 , 8     *  60)
hour12  = (24 * prediction_period * 12  , mt5.TIMEFRAME_H12, 12    *  60)
daily   = (24 * prediction_period * 24  , mt5.TIMEFRAME_D1 , 24    *  60)

delta_timeframe_pairs = [hourly, min5, min15, hour6, daily, hour2, hour3, hour4, hour6, hour8, hour12]
initial_thresh = 0.5

delta_timeframe_pair_pseudos = {
    'm'  : min5,
    'mm' : min15,
    'h'  : hourly,
    'h2' : hour2,
    'h3' : hour3,
    'h4' : hour4,
    'h6' : hour6,
    'h8' : hour8,
    'h12': hour12,
    'd'  : daily
}


eur_conv_pairs = {
    'GBP' : 'EURGBP',
    'USD' : 'EURUSD',
    'CHF' : 'EURCHF',
    'CAD' : 'EURCAD',
    'AUD' : 'EURAUD',
    'NZD' : 'EURNZD'
}


orders_list = [
    'buy_now',
    'sell_now',
    'buy',
    'sell',
    'buy_limit',
    'sell_limit',
    'buy_wide',
    'sell_wide',
    'buy_market',
    'sell_market',
    'buy_stop',
    'sell_stop'
]

symbols_list = [
    'EURUSD','EURGBP','EURCAD','EURAUD','EURCHF','EURNZD','EURSGD',
    'USDCHF','USDCAD','USDJPY','USDSGD',
    'GBPCHF','GBPAUD','GBPCAD','GBPNZD','GBPUSD','GBPJPY','GBPSGD',
    'NZDUSD', 'NZDCAD', 'NZDCHF','NZDJPY', 'NZDSGD',
    'AUDNZD', 'AUDCAD', 'AUDCHF','AUDUSD','AUDSGD','AUDJPY',
    'CADCHF','CADJPY',
    'BTCUSD','XRPUSD','ETHUSD',
    'eu', 'eg', 'ec', 'ea', 'ech', 'en','es',
    'uch', 'uc','uj','us',
    'gch', 'ga', 'gc', 'gn', 'gu','gj','gs',
    'nu','nc','nch','nj','ns'
    'an','ac','ach','au','as','aj'
    'cch','cj'
    'bu', 'xu', 'etu'
]
#34
demo = False

pseudos = {
    'eu' : 'EURUSD',
    'eg' : 'EURGBP',
    'ec' : 'EURCAD',
    'ea' : 'EURAUD',
    'ech': 'EURCHF',
    'en' : 'EURNZD',
    #'es' : 'EURSGD',
    'uch': 'USDCHF',
    'uc' : 'USDCAD',
    'uj' : 'USDJPY',
    #'us' : 'USDSGD',
    'gch': 'GBPCHF',
    'ga' : 'GBPAUD',
    'gc' : 'GBPCAD',
    'gn' : 'GBPNZD',
    'gu' : 'GBPUSD',
    'gj' : 'GBPJPY',
    #'gs' : 'GBPSGD',
    'nu' : 'NZDUSD',
    'nc' : 'NZDCAD',
    'nch': 'NZDCHF',
    'nj' : 'NZDJPY',
    #'ns' : 'NZDSGD',
    'an' : 'AUDNZD',
    'ac' : 'AUDCAD',
    'ach': 'AUDCHF',
    'au' : 'AUDUSD',
    #'as' : 'AUDSGD',
    'aj' : 'AUDJPY',
    'cch': 'CADCHF',
    'cj' : 'CADJPY',
    'bu' : f'BTCUSD{demo * ".bc"}',
    'etu': f'ETHUSD{demo * ".bc"}',
    'xu' : f'XRPUSD{demo * ".bc"}',
    'dou': f'DOGEUSD{demo * ".bc"}',
    'du' : f'DOTUSD{demo * ".bc"}',
    'adu': f'ADAUSD{demo * ".bc"}',
    'lu' : f'LTCUSD{demo * ".bc"}',
    'dau': f'DASHUSD{demo * ".bc"}',
    'xmu': f'XMRUSD{demo * ".bc"}',
    'neu': f'NEOUSD{demo * ".bc"}'
}

def symbol_converter(pseudo):
    if pseudo in pseudos:
        return pseudos[pseudo]
    else:
        return pseudo

# Définition du type d'ordre dans MetaTrader 5
order_types = {
    'buy': mt5.ORDER_TYPE_BUY,
    'sell': mt5.ORDER_TYPE_SELL,
    'buy_limit': mt5.ORDER_TYPE_BUY_LIMIT,
    'sell_limit': mt5.ORDER_TYPE_SELL_LIMIT,
    'buy_stop': mt5.ORDER_TYPE_BUY_STOP,
    'sell_stop': mt5.ORDER_TYPE_SELL_STOP
}

trading_styles = ['intraday', 'swing']

buy_orders  = ['buy', 'buy_limit', 'buy_now','buy_wide']
sell_orders = ['sell', 'sell_limit', 'sell_now', 'sell_wide']
markets = ['buy_market', 'sell_market']

period = 24 * prediction_period
testnum = 25
mean_period = 50
mperiod = 24 * mean_period
learning_rate = 0.0002
max_depth = 7
num_boost_round = 1000000
phases = ['train','test']
phase = "train"
percentile = 8
testnum_to_gain_ratio = 0.6


learning_trend = "bull"
trends = ["bull","bear"]

mode = "wide"

# (-0.8156619971583823, 1.0387544105866593) 3
# (-1.754685282490847, 2.024701438278825) 5
# (-1.3613752829088148, 1.609018819619437) 4
# (-1.1161564119504734, 1.3518697305903602) 3.5
# (-0.944350051201875, 1.1724213566700838) 3.2
# (-1.0329222428565212, 1.2651814128786065) 3.35
# (-1.2688076022019643, 1.5121681900643036) 3.8
narrow_factor = 0.2
bulk_factor = 3.0
narrowing_factors = [0.2, 0.6, 0.04]
bulking_factors = [0.12 , 0.36]
testnum_wide=15
testnum_narrow=ceil(testnum * narrow_factor)
testnum_short=ceil(testnum_narrow * narrow_factor)
testnum_inter=floor(testnum_short * bulk_factor)
testnum_bulk=floor(testnum_inter * bulk_factor)
test_data_path = "dtest.csv"


modes = ["wide", "narrow", "short","bulk","inter"]
folder = f"{learning_trend}_data"
modelfile_extension = ".json"
testfile_extension = ".csv"
extensions = [".json", ".ubj", ".bin", ".joblib", ".pkl"]
narfact = 1.0 # to choose on which time scale you which to perform  modelling and tests
model_in_use = f"model_data/M{prediction_period}_{prediction_period}_{learning_rate}_{percentile}_{learning_trend}_{mode}_{testnum * narfact}{modelfile_extension}"


