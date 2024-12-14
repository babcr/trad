import MetaTrader5 as mt5
from math import ceil
import numpy as np

#0,00560006
#0,00249453
#0,00555782
#0,00255166
#0,00533332
#0,00257822

def to_float(val):
    return np.float16(val)

dashboard = {
    'win_loss_quotient'         : 1.15,
    'equity_limit_ratio'        : 1.0,
    'goal'                      : 300,
    'risk_level'                : 6.0,
    'accepted_risk_overrun'     : 0.25,
    'min_offset_ratio'          : 0.001,
    'medium_low_offset_ratio'   : 0.02,
    'medium_high_offset_ratio'  : 0.05,
    'max_offset_ratio'          : 0.1,
    'stop_offset_ratio'         : 0.025,
    'loss_shrink_ratio'         : 0.1,
    'simple_to_volatile_ratio'  : 4,
    'defaultTradingMode'        : 'swing',
    'defaultDeltaTimeframePair' : 'h',
    'base_currency'             : 'EUR',
    #'bull_binary_wide_threshold'  : 0.56532, #  0.00020990
    #'bear_binary_wide_threshold'  : 0.51,   # 0.0
    'bull_binary_narrow_threshold': 0.583404798283354, # 0.007197525575728751
    'bull_binary_short_threshold' : 0.6147522934837489, # 0.007197384001368012
    'bull_binary_inter_threshold' : 0.6022845923473124, # 0.007197181178425822
    'bull_binary_bulk_threshold'  : 0.564222691309165, # 0.007196535577458332

    'bear_binary_narrow_threshold': 0.4774130875077242, # 0.007196824955709174
    'bear_binary_short_threshold' : 0.5147615199424868, # 0.007197727174090852
    'bear_binary_inter_threshold' : 0.4990051021701212, # 0.0071968194950124366
    'bear_binary_bulk_threshold'  : 0.45677515455850176, # 0.007196416933495793

    #'bull_binary_wide_comb'  : 0.565184951930333,
    #'bear_binary_wide_comb'  : 0.5,   # 0.0
    'bull_binary_narrow_comb': 0.5530381813900536, # 0.11338179460910383
    'bull_binary_short_comb' : 0.5400545615244122, # 0.11338198806861995
    'bull_binary_inter_comb' : 0.549506785268383, # 0.11338177439040224
    'bull_binary_bulk_comb'  : 0.5529538988438077, # 0.11338125020465635

    'bear_binary_narrow_comb': 0.45339694696911087, # 0.11338190850036475
    'bear_binary_short_comb' : 0.494469045250951, # 0.11338006069326532
    'bear_binary_inter_comb' : 0.47345983847696615, # 0.11338102030902121
    'bear_binary_bulk_comb'  : 0.4317358144988849 # 0.11337965061913803
}

dashboard2 = {
    'bull_binary_narrow_threshold': 0.5815593672108603, # 0.00883
    'bull_binary_short_threshold' : 0.6080526976936248, # 0.00883
    'bull_binary_inter_threshold' : 0.5986122660162314, # 0.00883
    'bull_binary_bulk_threshold'  : 0.5635191578539142, # 0.00883

    'bear_binary_narrow_threshold': 0.47546789825352226, # 0.00883 # 0.47017644667952435, # 0.017821
    'bear_binary_short_threshold' : 0.513359848438987, # 0.00883 # 0.5090152800705919, # 0.017821
    'bear_binary_inter_threshold' : 0.4970527687143775, # 0.00883 # 0.4915199141875779, # 0.017821
    'bear_binary_bulk_threshold'  : 0.4544354476166901, # 0.00883 #0.4487391455873384, # 0.017821

    'bull_binary_narrow_comb': 0.5511242783610621, # 0.12751
    'bull_binary_short_comb' : 0.5374584301933921, # 0.12751
    'bull_binary_inter_comb' : 0.5469676956981749, # 0.12751
    'bull_binary_bulk_comb'  : 0.5521594874757811, # 0.12751

    'bear_binary_narrow_comb': 0.4510338879865221, # 0.12751 # 0.4452927478811478, # 0.195588
    'bear_binary_short_comb' : 0.4928303303471458, # 0.12751 # 0.48780455645129395, # 0.195589
    'bear_binary_inter_comb' : 0.4710052883619704, # 0.12751
    'bear_binary_bulk_comb'  : 0.4291395023153211 # 0.12751 # 0.42340164752977744 # 0.195588
}
data_generator_categorizing_threshold = 1.5
certitude_degree_of_categorization = 1.15
spreads = []
deviation = 5
limit_spread = 0.03
max_spread = 0.1
limit_correlation = 0.3
unfilled_order_lifespan_min = 5


prediction_period = 30 # in days
minutes_in_a_year = 525600
hours_before_repeat_order = 5

mean_period = 50
mperiod = 24 * mean_period

hourly  = (24 * mean_period, mt5.TIMEFRAME_H1)
min5    = (2 * mean_period, mt5.TIMEFRAME_M5)
min15   = (6 * mean_period, mt5.TIMEFRAME_M15)
hour6   = (24 * 6 * mean_period, mt5.TIMEFRAME_H6)
daily   = (24 * 24 * mean_period, mt5.TIMEFRAME_D1)

delta_timeframe_pairs = [hourly, min5, min15, hour6, daily]

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
demo = True
#34
pseudos = {
    'eu' : 'EURUSD',
    'eg' : 'EURGBP',
    'ec' : 'EURCAD',
    'ea' : 'EURAUD',
    'ech': 'EURCHF',
    'en' : 'EURNZD',
    'es' : 'EURSGD',

    'uch': 'USDCHF',
    'uc' : 'USDCAD',
    'uj' : 'USDJPY',
    'us' : 'USDSGD',

    'gch': 'GBPCHF',
    'ga' : 'GBPAUD',
    'gc' : 'GBPCAD',
    'gn' : 'GBPNZD',
    'gu' : 'GBPUSD',
    'gj' : 'GBPJPY',
    'gs' : 'GBPSGD',

    'nu' : 'NZDUSD',
    'nc' : 'NZDCAD',
    'nch': 'NZDCHF',
    'nj' : 'NZDJPY',
    'ns' : 'NZDSGD',

    'an' : 'AUDNZD',
    'ac' : 'AUDCAD',
    'ach': 'AUDCHF',
    'au' : 'AUDUSD',
    'as' : 'AUDSGD',
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

trading_styles = ['intraday', 'swing']

buy_orders  = ['buy', 'buy_limit', 'buy_now','buy_wide']
sell_orders = ['sell', 'sell_limit', 'sell_now', 'sell_wide']
markets = ['buy_market', 'sell_market']
order_types_ = ['buy_market', 'sell_market']

period=24 * prediction_period

testnum=25



learning_rate_1 = 0.001
learning_rate = 0.0001
# act_threshold15_50_1.5_0.001.json
model_in_use_wide_bull = f"M15_50_0.0001_2_bull_wide_25.0.json"
model_in_use_wide_bear = f"M15_50_0.0001_2_bear_wide_25.0.json"
model_in_use_narrow_bull = f"M15_50_0.0002_2_bull_narrow_5.0.json"
model_in_use_narrow_bear = f"M15_50_0.0002_2_bear_narrow_5.0.json"
model_in_use_short_bull = f"M15_50_0.0003_2_bull_short_1.0.json"
model_in_use_short_bear = f"M15_50_0.0003_2_bear_short_1.0.json"

model_in_use_inter_bull = f"M15_50_0.0003_2_bull_inter_3.0.json"
model_in_use_inter_bear = f"M15_50_0.0003_2_bear_inter_3.0.json"
model_in_use_bulk_bull = f"M15_50_0.0002_2_bull_bulk_9.0.json"
model_in_use_bulk_bear = f"M15_50_0.0002_2_bear_bulk_9.0.json"
