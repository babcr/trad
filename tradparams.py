import MetaTrader5 as mt5
from math import ceil, floor
import numpy as np
import xgboost as xgb

def to_float(val):
    return np.float16(val)

deviation                   = 5

win_loss_quotient           = 0.5
equity_limit_ratio          = 0.05
goal                        = 1.1
risk_level                  = 20.0
accepted_risk_overrun       = 0.25
limit_spread                = 0.01
max_spread                  = 0.15
limit_correlation           = 0.2

loss_shrink_ratio           = 2.0
offset_shrink_ratio         = 0.003
granularity_factor          = 10
defaultTradingMode          = 'swing'

unfilled_order_lifespan_min = 5
prediction_period           = 30 # in days
hours_before_repeat_order   = 24
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

prob_limits_h12 = {
    'bull_binary_short_threshold'       : 0.5392410706456066, # 0.5332410706456066, # 0,00244735 63.15789474 %
    'bull_binary_short_rev_threshold'   : 0.5392410706456066, # 0.5332410706456066, # 0,00244735 63.15789474 %
    'bull_binary_short_comb'            : 0.53608246678862,   # 0.53108246678862, # 0,00598957 62.36559140 %
    'bull_binary_short_rev_comb'        : 0.53608246678862,   # 0.53108246678862, # 0,00598957 62.36559140 %
    
    'bear_binary_short_threshold'       : 0.5224809636477462, # 0.5124809636477462, # 0.00148244 53.26086957 %
    'bear_binary_short_comb'            : 0.5122558303311716, # 0.00157912 54.08163265 % 0.5096337922503382, # 0.00391557 51.85185185 %
}

prob_limits_h4 = {
    'bull_binary_short_threshold'       : 0.6106149845718837, # 0.6041442694063516, # 0.00118720 80.44444444 %
    'bull_binary_short_rev_threshold'   : 0.6106149845718837, # 0.6041442694063516, # 0.00118720 80.44444444 %
    'bull_binary_short_comb'            : 0.6019094306049278, # 0.5961690525666233, # 0.00326084 77.34627832 %
    'bull_binary_short_rev_comb'        : 0.6019094306049278, # 0.5961690525666233, # 0.00326084 77.34627832 %

    'bear_binary_short_threshold'       : 0.5441958523248068, # 0.0007 52.69230769 % 0.5370415609803433, # 0.5270415609803433, # 0,00136903 58.13953488 %
    'bear_binary_short_comb'            : 0.5366944232231559  # 0.00137187 52.69230769 % 0.5248195848692672  # 0.00203763 53.12500000 %
}

prob_limits_h3 = {
    'bull_binary_narrow_threshold'      : 0.5,
    'bull_binary_narrow_rev_threshold'  : 0.5,
    'bull_binary_narrow_comb'           : 0.5,
    'bull_binary_narrow_rev_comb'       : 0.5,

    'bear_binary_narrow_threshold'      : 0.5,
    'bear_binary_narrow_comb'           : 0.5,


    'bull_binary_inter_threshold'       : 0.5,
    'bull_binary_inter_rev_threshold'   : 0.5,
    'bull_binary_inter_comb'            : 0.5,
    'bull_binary_inter_rev_comb'        : 0.5,

    'bear_binary_inter_threshold'       : 0.5,
    'bear_binary_inter_comb'            : 0.5, 
}


prob_limits_h8 = {
    'bull_binary_short_threshold'       : 0.6085905684546528, # 0.6080905684546528, # 0.00151824 62.23776224 %
    'bull_binary_short_rev_threshold'   : 0.6085905684546528, # 0.00151824 62.23776224 % 
    'bull_binary_short_comb'            : 0.6080905684546528, # 0.00241007 66.96035242 % 
    'bull_binary_short_rev_comb'        : 0.6080905684546528, # 0.00241007 66.96035242 % 

    'bear_binary_short_threshold'       : 0.5277568889405015, # 0.00139084 53.43511450 %
    'bear_binary_short_comb'            : 0.5255484240169545  # 0.00234637 51.58371041 %
}

prob_limits_h6 = {
    'bull_binary_short_threshold'       : 0.6091442694063516, # 0.6041442694063516, # 0,00194994 70.96774194 %
    'bull_binary_short_rev_threshold'   : 0.6091442694063516, # 0.6041442694063516, # 0,00194994 70.96774194 %
    'bull_binary_short_comb'            : 0.6011690525666233, # 0.5961690525666233, # 0,0048765 70.26476578 %
    'bull_binary_short_rev_comb'        : 0.6011690525666233, # 0.5961690525666233, # 0,0048765 70.26476578 %

    'bear_binary_short_threshold'       : 0.5370415609803433, # 0.5270415609803433, # 0,00136903 58.13953488 %
    'bear_binary_short_comb'            : 0.5248195848692672  # 0.00203763 53.12500000 %
}

prob_limits_h = {
    'bull_binary_short_threshold'       : 0.5857174235645772, # 0.5807174235645772, # 0,00154621 82.30215827 %
    'bull_binary_short_rev_threshold'   : 0.5857174235645772, # 0.5807174235645772, # 0,00154621 82.30215827 %
    'bull_binary_short_comb'            : 0.5716092954776367,  # 0.5666092954776367,  # 0,00524377 78.19261773 %
    'bull_binary_short_rev_comb'        : 0.5716092954776367,  # 0.5666092954776367,  # 0,00524377 78.19261773 %

    'bear_binary_short_threshold'       : 0.5276190522977312, # 0.5226190522977312, # 0,00095217 57.94392523 %
    'bear_binary_short_comb'            : 0.5236698965611963  # 0.5186698965611963  # 0,00476085 54.57943925 %
} 

prob_limits_mm = {
    'bull_binary_short_threshold'       : 0.5246555490752431, # 0.5196555490752431, # 0,00098558 60.88709677 %
    'bull_binary_short_rev_threshold'   : 0.5246555490752431, # 0.5196555490752431, # 0,00098558 60.88709677 %
    'bull_binary_short_comb'            : 0.5226691492884165, # 0.5176691492884165, # 0,00492392 59.24132365 %
    'bull_binary_short_rev_comb'        : 0.5226691492884165, # 0.5176691492884165, # 0,00492392 59.24132365 %

    'bear_binary_short_threshold'       : 0.513721826181881, # 0.508721826181881, # 0,00184796 53.76344086 %
    'bear_binary_short_comb'            : 0.5122449507474112  # 0.5072449507474112  # 0,00462586 54.03780069 %
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
    'learning_rates'    ,
    'preds'             ,
    'models'            ,
    'prob_limits'       ,
    'candles'           ,
    'scores'            ,
    'base_sum_supports' ,
    'base_sums'
]

special_percs = {
    #'mm' : {'prob_limits' : prob_limits_mm },
	#'h'  : {'prob_limits' : prob_limits_h  },
    'h3' : {
        'prob_limits' : prob_limits_h3 , 
        'learning_rates' : {
            'inter' : 0.2, 
            'narrow' : 0.1
        }
    },
    #'h4' : {'prob_limits' : prob_limits_h4 },
	#'h6' : {'prob_limits' : prob_limits_h6 },
    #'h8' : {'prob_limits' : prob_limits_h8 },
    #'h12': {'prob_limits' : prob_limits_h12}
}

used_timeframes = {
    #'mm' : min15 ,
    #'h'  : hourly,
    'h3' : hour3 ,
    #'h4' : hour4 ,
    
    #'h6' : hour6 ,
    #'h8' : hour8 ,
    #'h12': hour12
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
    'buy_now'    ,
    'sell_now'   ,
    'buy'        ,
    'sell'       ,
    'buy_limit'  ,
    'sell_limit' ,
    'buy_wide'   ,
    'sell_wide'  ,
    'buy_market' ,
    'sell_market',
    'buy_stop'   ,
    'sell_stop'
]

symbols_list = [
    'EURUSD','EURGBP' ,'EURCAD' ,'EURAUD','EURCHF' ,'EURNZD','EURSGD',
    'USDCHF','USDCAD' ,'USDJPY' ,'USDSGD',
    'GBPCHF','GBPAUD' ,'GBPCAD' ,'GBPNZD','GBPUSD' ,'GBPJPY','GBPSGD',
    'NZDUSD', 'NZDCAD', 'NZDCHF','NZDJPY', 'NZDSGD',
    'AUDNZD', 'AUDCAD', 'AUDCHF','AUDUSD','AUDSGD' ,'AUDJPY',
    'CADCHF','CADJPY' ,
    'BTCUSD','XRPUSD' ,'ETHUSD' ,
    'eu' , 'eg', 'ec', 'ea', 'ech', 'en','es',
    'uch', 'uc','uj' ,'us' ,
    'gch', 'ga', 'gc', 'gn', 'gu' ,'gj' ,'gs',
    'nu','nc','nch','nj','ns',
    'an','ac','ach','au','as','aj',
    'cch','cj' ,
    'bu' , 'xu', 'etu', 'dou', 'du', 'adu', 'lu', 'dau', 'xmu', 'neu'
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
learning_rate = 0.1
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
    #'bulk',
    'narrow',
    'inter',
    #'short'
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



