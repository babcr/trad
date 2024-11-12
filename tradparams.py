import MetaTrader5 as mt5

dashboard = {
    'win_loss_quotient'         : 1.25,
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

data_generator_categorizing_threshold = 1.5
certitude_degree_of_categorization = 1.15


prediction_period = 15 # in days
minutes_in_a_year = 525600
hourly  = (24 * prediction_period * 2, mt5.TIMEFRAME_H1)
min5    = (2 * prediction_period * 2, mt5.TIMEFRAME_M5)
min15   = (6 * prediction_period * 2, mt5.TIMEFRAME_M15)
hour6   = (24 * 6 * prediction_period * 2, mt5.TIMEFRAME_H6)
daily   = (24 * 24 * prediction_period * 2, mt5.TIMEFRAME_D1)

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
    'EURUSD','EURGBP','EURCAD','EURAUD','EURCHF','EURNZD',
    'USDCHF','USDCAD','USDAUD','USDNZD',
    'GBPCHF','GBPAUD','GBPCAD','GBPNZD','GBPUSD',
    'BTCUSD','XRPUSD','ETHUSD',
    'eu', 'eg', 'ec', 'ea', 'ech', 'en', 
    'uch', 'uc', 'ua', 'un',
    'gch', 'ga', 'gc', 'gn', 'gu', 
    'bu', 'xu', 'etu'
]

pseudos = {
    'eu' : 'EURUSD',
    'eg' : 'EURGBP',
    'ec' : 'EURCAD',
    'ea' : 'EURAUD',
    'ech': 'EURCHF',
    'en' : 'EURNZD',

    'uch': 'USDCHF',
    'uc' : 'USDCAD',

    'gch': 'GBPCHF',
    'ga' : 'GBPAUD',
    'gc' : 'GBPCAD',
    'gn' : 'GBPNZD',
    'gu' : 'GBPUSD',

    'bu' : 'BTCUSD.bc',
    'xu' : 'XRPUSD.bc',
    'etu': 'ETHUSD.bc'   
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

period=24 * prediction_period

testnum=25

mean_period = 50
mperiod = 24 * mean_period

learning_rate_1 = 0.001
learning_rate = 0.0001
# act_threshold15_50_1.5_0.001.json
model_in_use = f"validated/act_threshold{prediction_period}_{mean_period}_{data_generator_categorizing_threshold}_{learning_rate_1}.json"
