import MetaTrader5 as mt5
from math import ceil, floor
import numpy as np

def to_float(val):
    return np.float16(val)

deviation                   = 5

win_loss_quotient           = 4.15
equity_limit_ratio          = 0.1
goal                        = 1.1
risk_level                  = 10.0
accepted_risk_overrun       = 0.25
limit_spread                = 0.03
max_spread                  = 0.2
limit_correlation           = 0.5

loss_shrink_ratio           = 0.3
offset_shrink_ratio         = 0.005
granularity_factor          = 10
defaultTradingMode          = 'swing'

unfilled_order_lifespan_min = 5
prediction_period           = 30 # in days
hours_before_repeat_order   = 5
mean_period                 = 50

dashboard                   = {
    # Risk management
    'win_loss_quotient'                 : win_loss_quotient,
    'equity_limit_ratio'                : equity_limit_ratio,
    'goal'                              : goal,
    'risk_level'                        : risk_level,
    'accepted_risk_overrun'             : accepted_risk_overrun,
    'limit_correlation'                 : limit_correlation,
    'limit_spread'                      : limit_spread,
    'max_spread'                        : max_spread,

    # Volatility management
    'loss_shrink_ratio'                 : loss_shrink_ratio,
    'offset_shrink_ratio'               : offset_shrink_ratio,
    'granularity_factor'                : granularity_factor,
    'defaultTradingMode'                : defaultTradingMode,

    # Base parameters
    'defaultDeltaTimeframePair'         : 'h',
    'base_currency'                     : 'EUR',

    'unfilled_order_lifespan_min'       : unfilled_order_lifespan_min,
    'prediction_period'                 : prediction_period, # in days
    'hours_before_repeat_order'         : hours_before_repeat_order,
    'mean_period'                       : mean_period,

    # Models parameters
    'bull_binary_wide_threshold'        : 0.5410273671465529, #  0.00959177 # 51.05457708 %
    'bull_binary_bulk_threshold'        : 0.5601240442207156, # 0.00966633 # 58.28928918 %
    'bull_binary_narrow_threshold'      : 0.5864701454549608, # 0.00974751 # 69.47593583 %
    'bull_binary_inter_threshold'       : 0.6608582432204393, # 0.00294289 # 83.59900815 %
    'bull_binary_short_threshold'       : 0.669481696448141, # 0.00296983 # 87.32888733 %

    'bull_binary_wide_rev_threshold'    : 0.5410273671465529, #  0.00959177 # 51.05457708 %
    'bull_binary_bulk_rev_threshold'    : 0.5601240442207156, # 0.00966633 # 58.28928918 %
    'bull_binary_narrow_rev_threshold'  : 0.5864701454549608, # 0.00974751 # 69.47593583 %
    'bull_binary_inter_rev_threshold'   : 0.6608582432204393, # 0.00294289 # 83.59900815 %
    'bull_binary_short_rev_threshold'   : 0.669481696448141, # 0.00296983 # 87.32888733 %

    'bear_binary_wide_threshold'        : 0.55, # 0.0
    'bear_binary_bulk_threshold'        : 0.45818963200061946, # 0.00304660 51.64271047 %
    'bear_binary_narrow_threshold'      : 0.55, # 0.0
    'bear_binary_inter_threshold'       : 0.55, # 0.0
    'bear_binary_short_threshold'       : 0.5234204732099372, # 0.00180324 # 56.03000577 %

    'bull_binary_wide_comb'             : 0.5410273671465529, # 0.00959177 # 4.30919591 %
    'bull_binary_bulk_comb'             : 0.5443996788063195, # 0.17400235 # 54.79788598 %
    'bull_binary_narrow_comb'           : 0.5426327002956575, # 0.17544586 # 57.59606871 %
    'bull_binary_inter_comb'            : 0.5388572239431149, # 0.17655049 # 57.97718443 %
    'bull_binary_short_comb'            : 0.5250615484040865, # 0.17814728 # 56.26422624 %

    'bull_binary_wide_rev_comb'         : 0.5410273671465529, # 0.00959177 # 4.30919591 %
    'bull_binary_bulk_rev_comb'         : 0.5474969152440783, # 0.11600018 # 25.92803184 %
    'bull_binary_narrow_rev_comb'       : 0.5505109433060268, # 0.11696391 # 49.17049770 %
    'bull_binary_inter_rev_comb'        : 0.5499325410510577, # 0.11770102 # 60.73482207 %
    'bull_binary_short_rev_comb'        : 0.669481696448141, # 0.02969208 # 73.82741188 %

    'bear_binary_wide_comb'             : 0.55, # 0.0
    'bear_binary_bulk_comb'             : 0.45818963200061946, # 0.00304660 51.64271047 %
    'bear_binary_narrow_comb'           : 0.55, # 0.0
    'bear_binary_inter_comb'            : 0.55, # 0.0
    'bear_binary_short_comb'            : 0.5142886711656096, # 0.00902938 # 53.80974371 %
}

'''
'bull_binary_wide_comb'         : 0.5410273671465529, # 0.00959177 # 4.30919591 %
'bull_binary_bulk_comb'         : 0.540433049216802, # 0.27066881 53.56183961 % 0.538786590245818, # 0.31900361 # 53.10746281 % 
'bull_binary_narrow_comb'       : 0.5379120127425855, # 0.22418134 56.23863578 % 0.5370921499261588, # 0.23392781 # 56.00254916 % 
'bull_binary_inter_comb'        : 0.5388572239431149, # 0.17655049 57.97718443 % 0.5456200863884992, # 0.13731821 # 60.18720962 % 
'bull_binary_short_comb'        : 0.571667379499238, # 0.02969208 # 73.82741188 % 0.5417637841197652, # 0.07917645 # 62.56467645 % # 0.5538775951696272, # 0.04948541 # 67.88843950 %  0.5417637841197652, # 0.07917645 # 62.56467645 %
'''
minutes_in_a_year = 525600

hourly  = (24 * mean_period, mt5.TIMEFRAME_H1)
min5    = (2 * mean_period, mt5.TIMEFRAME_M5)
min15   = (6 * mean_period, mt5.TIMEFRAME_M15)
hour6   = (24 * 6 * mean_period, mt5.TIMEFRAME_H6)
daily   = (24 * 24 * mean_period, mt5.TIMEFRAME_D1)

delta_timeframe_pairs = [hourly, min5, min15, hour6, daily]
initial_thresh = 0.5

delta_timeframe_pair_pseudos = {
    'h' : hourly,
    'm' : min5,
    'mm': min15,
    's' : hour6,
    'd' : daily
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
demo = False
#34
#29
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
    'xu' : f'XRPUSD{demo * ".bc"}'
}
pseudos_we = {
    'bu' : f'BTCUSD{demo * ".bc"}',
    'xu' : f'XRPUSD{demo * ".bc"}',
    'etu': f'ETHUSD{demo * ".bc"}'
}
volatiles = {
    f'BTCUSD{demo * ".bc"}',
    f'XRPUSD{demo * ".bc"}',
    f'ETHUSD{demo * ".bc"}'
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

order_suffix = ''
trading_styles = ['intraday', 'swing']

buy_orders  = ['buy', 'buy_limit', 'buy_now','buy_wide']
sell_orders = ['sell', 'sell_limit', 'sell_now', 'sell_wide']
markets = ['buy_market', 'sell_market']
order_types_ = ['buy_market', 'sell_market']

period = 24 * prediction_period
mperiod = 24 * mean_period

testnum = 25
learning_rate = 0.0002
max_depth = 7
num_boost_round = 1000000
phases = ['train','test']
phase = "train"
percentile = 2
learning_trend = "bull"
trends = ["bull","bear"]
mode = "wide"

# act_threshold15_50_1.5_0.001.json
model_in_use_wide_bull = f"M30_50_0.0003_2_bull_wide_25.0.json"
model_in_use_wide_bear = f"M30_50_0.0003_2_bear_wide_25.0.json"
model_in_use_narrow_bull = f"M30_50_0.0003_2_bull_narrow_5.0.json"
model_in_use_narrow_bear = f"M30_50_0.0003_2_bear_narrow_5.0.json"
model_in_use_short_bull = f"M30_50_0.0003_2_bull_short_1.0.json"
model_in_use_short_bear = f"M30_50_0.0003_2_bear_short_1.0.json"
model_in_use_inter_bull = f"M30_50_0.0003_2_bull_inter_3.0.json"
model_in_use_inter_bear = f"M30_50_0.0003_2_bear_inter_3.0.json"
model_in_use_bulk_bull = f"M30_50_0.0003_2_bull_bulk_9.0.json"
model_in_use_bulk_bear = f"M30_50_0.0003_2_bear_bulk_9.0.json"

# [91841] train-logloss:0.64667   eval-logloss:0.68570  M30_50_0.00005_2_bear_inter_3.0.json 1 205 704
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
model_in_use = f"M{prediction_period}_{mean_period}_{learning_rate}_{percentile}_{learning_trend}_{mode}_{testnum * narfact}{modelfile_extension}"