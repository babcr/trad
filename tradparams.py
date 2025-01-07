import MetaTrader5 as mt5
from math import ceil, floor
import numpy as np
import xgboost as xgb

def to_float(val):
    return np.float16(val)

deviation                   = 5

win_loss_quotient           = 0.5
equity_limit_ratio          = 0.1
goal                        = 1.1
risk_level                  = 20.0
accepted_risk_overrun       = 0.25
limit_spread                = 0.03
max_spread                  = 0.15
limit_correlation           = 0.2

loss_shrink_ratio           = 2.0
offset_shrink_ratio         = 0.003
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
    'mean_period'                       : mean_period
}

prob_limits_h = {
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
    'bull_binary_short_rev_comb'        : 0.5330840969945617, # 0.11876520 # 43.55586909 % 0.669481696448141, # 0.02969208 # 73.82741188 %

    'bear_binary_wide_comb'             : 0.55, # 0.0
    'bear_binary_bulk_comb'             : 0.45818963200061946, # 0.00304660 51.64271047 %
    'bear_binary_narrow_comb'           : 0.55, # 0.0
    'bear_binary_inter_comb'            : 0.55, # 0.0
    'bear_binary_short_comb'            : 0.5142886711656096 # 0.00902938 # 53.80974371 %
}

prob_limits_mm = {
    # Models parameters
    'bull_binary_wide_threshold'        : 0.49854904516036763,# 0.00204688 63.74611935 %
    'bull_binary_bulk_threshold'        : 0.4976227593850317, # 0.00203893 63.91966759 %
    'bull_binary_narrow_threshold'      : 0.49915087102411126,# 0.00101376 72.56267409 % 0.5864701454549608, # 0.00974751 69.47593583 %
    'bull_binary_inter_threshold'       : 0.5244010163925317, # 0.00101373 77.50696379 % 0.6608582432204393, # 0.00294289 83.59900815 %
    'bull_binary_short_threshold'       : 0.5288713666589832, # 0.00100453 80.74490513 %

    'bull_binary_wide_rev_threshold'    : 0.49854904516036763,# 0.00204688 63.74611935 %
    'bull_binary_bulk_rev_threshold'    : 0.4976227593850317, # 0.00203893 63.91966759 %
    'bull_binary_narrow_rev_threshold'  : 0.49915087102411126,# 0.00101376 72.56267409 % 0.5864701454549608, # 0.00974751 69.47593583 %
    'bull_binary_inter_rev_threshold'   : 0.5244010163925317, # 0.00101373 77.50696379 % 0.6608582432204393, # 0.00294289 83.59900815 %
    'bull_binary_short_rev_threshold'   : 0.5288713666589832, # 0.00100453 80.74490513 %

    'bear_binary_wide_threshold'        : 0.55, # 0.0
    'bear_binary_bulk_threshold'        : 0.55, # 0.00304660 51.64271047 %
    'bear_binary_narrow_threshold'      : 0.55, # 0.0
    'bear_binary_inter_threshold'       : 0.5015384924332926, # 0.00001977 57.14285714 %
    'bear_binary_short_threshold'       : 0.5234204732099372, # 0.00009318 59.09090909 %

    'bull_binary_wide_comb'             : 0.4979320762750071, # 0.12277105 55.31196623 %
    'bull_binary_bulk_comb'             : 0.4968913861987393, # 0.12230493 55.35165901 %
    'bull_binary_narrow_comb'           : 0.4969840647057671, # 0.12180755 56.55003738 %
    'bull_binary_inter_comb'            : 0.5048632626815343, # 0.12152738 58.07585289 % 0.5388572239431149, # 0.17655049 57.97718443 %
    'bull_binary_short_comb'            : 0.5092061166097849, # 0.12044072 54.92717522 % 0.5132003437379862, # 0.05018463 58.14097424 %

    'bull_binary_wide_rev_comb'         : 0.4979320762750071, # 0.12277105 55.31196623 % 0.4980866612480027, # 0.05115854 57.46660042 % 0.4979320762750071, # 0.12277105 55.31196623 %
    'bull_binary_bulk_rev_comb'         : 0.4968913861987393, # 0.12230493 55.35165901 % 0.49707299459987403,# 0.05095144 57.84062409 % 0.4968913861987393, # 0.12230493 55.35165901 %
    'bull_binary_narrow_rev_comb'       : 0.4969840647057671, # 0.12180755 56.55003738 % 0.4974497259699691, # 0.05074885 60.16748741 % 0.4969840647057671, # 0.12180755 56.55003738 %
    'bull_binary_inter_rev_comb'        : 0.5048632626815343, # 0.12152738 58.07585289 % 0.509151161171405,  # 0.05063658 63.21710884 % 0.5048632626815343, # 0.12152738 58.07585289 % 0.5388572239431149, # 0.17655049 57.97718443 %
    'bull_binary_short_rev_comb'        : 0.5092061166097849, # 0.12044072 54.92717522 % 0.5132003437379862, # 0.05018463 58.14097424 % 0.5092061166097849, # 0.12044072 54.92717522 % 0.5132003437379862, # 0.05018463 58.14097424 %

    'bear_binary_wide_comb'             : 0.55, # 0.0
    'bear_binary_bulk_comb'             : 0.55, # 0.00304660 51.64271047 %
    'bear_binary_narrow_comb'           : 0.55, # 0.0
    'bear_binary_inter_comb'            : 0.5015384924332926, # 0.00001977 57.14285714 %
    'bear_binary_short_comb'            : 0.5128124231821756  # 0.00027813 56.09137056 %
}

initial_preds = {
    "pred_bull_bulk"        : 0,
    "pred_bull_narrow"      : 0,
    "pred_bull_inter"       : 0,
    "pred_bull_short"       : 0,
    "pred_bear_bulk"        : 0,
    "pred_bear_narrow"      : 0,
    "pred_bear_inter"       : 0,
    "pred_bear_short"       : 0,
    "pred_narrow"           : 0,
    "pred_short"            : 0,
    "pred_inter"            : 0,
    "pred_bulk"             : 0,
    "pred_narrow_"          : 0,
    "pred_short_"           : 0,
    "pred_inter_"           : 0,
    "pred_bulk_"            : 0,
    "pred_rev_bear_bulk"    : 0,
    "pred_rev_bear_narrow"  : 0,
    "pred_rev_bear_inter"   : 0,
    "pred_rev_bear_short"   : 0,
    "pred_rev_bull_bulk"    : 0,
    "pred_rev_bull_narrow"  : 0,
    "pred_rev_bull_inter"   : 0,
    "pred_rev_bull_short"   : 0,
    "pred_rev_narrow"       : 0,
    "pred_rev_short"        : 0,
    "pred_rev_inter"        : 0,
    "pred_rev_bulk"         : 0,
    "pred_rev_narrow_"      : 0,
    "pred_rev_short_"       : 0,
    "pred_rev_inter_"       : 0,
    "pred_rev_bulk_"        : 0,
    "pred"                  : 1,
    "pred_"                 : 1,
    "pred_rev"              : 1,
    "pred_rev_"             : 1
}

minutes_in_a_year = 525600

min5    = (24 * mean_period / 12 , mt5.TIMEFRAME_M5 , 1/12*  60)
min15   = (24 * mean_period / 4  , mt5.TIMEFRAME_M15, 0.25*  60)
hourly  = (24 * mean_period      , mt5.TIMEFRAME_H1 , 1   *  60)
hour2   = (24 * mean_period * 2  , mt5.TIMEFRAME_H2 , 2   *  60)
hour3   = (24 * mean_period * 3  , mt5.TIMEFRAME_H3 , 3   *  60)
hour4   = (24 * mean_period * 4  , mt5.TIMEFRAME_H4 , 4   *  60)
hour6   = (24 * mean_period * 6  , mt5.TIMEFRAME_H6 , 6   *  60)
hour8   = (24 * mean_period * 8  , mt5.TIMEFRAME_H8 , 8   *  60)
hour12  = (24 * mean_period * 12 , mt5.TIMEFRAME_H12, 12  *  60)
daily   = (24 * mean_period * 24 , mt5.TIMEFRAME_D1 , 24  *  60)

delta_timeframe_pairs = [
    hourly,
    min5,
    min15,
    hour2,
    hour3,
    hour4,
    hour6,
    hour8,
    hour12,
    daily
]

initial_thresh = 0.5

delta_timeframe_pair_pseudos = {
    'm'  : min5     ,
    'mm' : min15    ,
    'h'  : hourly   ,
    'h2' : hour2    ,
    'h3' : hour3    ,
    'h4' : hour4    ,
    'h6' : hour6    ,
    'h8' : hour8    ,
    'h12': hour12   ,
    'd'  : daily
}

percs = [
    'preds'             ,
    'models'            ,
    'prob_limits'       ,
    'candles'           ,
    'scores'            ,
    'base_sum_supports' ,
    'base_sums'
]

special_percs = {
	'h' : {'prob_limits' : prob_limits_h},
	'mm': {'prob_limits' : prob_limits_mm}
}

used_timeframes = {
    'mm' : min15,
    'h'  : hourly
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
    'bu', 'xu', 'etu', 'dou', 'du', 'adu', 'lu', 'dau', 'xmu', 'neu'
]
demo = False
#34
#29
#36
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


pseudos_we = {
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
    'buy'       : mt5.ORDER_TYPE_BUY,
    'sell'      : mt5.ORDER_TYPE_SELL,
    'buy_limit' : mt5.ORDER_TYPE_BUY_LIMIT,
    'sell_limit': mt5.ORDER_TYPE_SELL_LIMIT,
    'buy_stop'  : mt5.ORDER_TYPE_BUY_STOP,
    'sell_stop' : mt5.ORDER_TYPE_SELL_STOP
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
learning_rate = 0.0003
max_depth = 7
num_boost_round = 1000000
phases = ['train','test']
phase = "train"
percentile = 2
learning_trend = "bull"
trends = ["bull","bear"]
mode = "wide"

modes = [
    'comb',
    'threshold'
]

directions = [
    'bull',
    'bear'
]
ranges = [
    'bulk',
    'narrow',
    'inter',
    'short'
]

ranges_equi = {
    'bulk' : '9.0',
    'narrow' : '5.0',
    'inter' : '3.0',
    'short' : '1.0'   
}

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

folder = f"{learning_trend}_data"
modelfile_extension = ".json"
testfile_extension = ".csv"
extensions = [".json", ".ubj", ".bin", ".joblib", ".pkl"]
narfact = 1.0 # to choose on which time scale you which to perform  modelling and tests
model_in_use = f"M{prediction_period}_{mean_period}_{learning_rate}_{percentile}_{learning_trend}_{mode}_{testnum * narfact}{modelfile_extension}"



