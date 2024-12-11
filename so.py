#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tradautotools as ta
import argparse



def convert_to_boolean(value):
    return value.lower() in ['true', '1', 'yes', 'y']

def main(
        symbol     ,
        ordertype  ,
        volume     ,
        price      ,
        mode       ,
        delta_timeframe_pair
    ):

    order_type, vol, p, sl, tp, typefilling = ta.get_attributes(
        symbol     ,
        ordertype  ,
        volume     ,
        price      ,
        mode       ,
        delta_timeframe_pair
    )
    print(f"Account Expendable Equity = {ta.get_equity()}")

    ta.send_order(symbol=symbol, order_type=order_type, volume=vol, price=p, stoploss=sl, takeprofit=tp, typefilling=typefilling)

    print(f"\nAttributes = ({order_type}, {vol}, {p}, {sl}, {tp})")
    
    print(f"Symbol {symbol} ( max, average ) candle sizes : {ta.candle_size(symbol, delta_timeframe_pair)}")

    # print(f"Symbol {symbol} conversion factor (conversion pair is {ta.get_conv_pair(symbol)}): {ta.get_conversion_factor(symbol)}")

    print(f"Symbol {symbol} lot size is {ta.get_contract_size(symbol)}")

    print(f"Symbol {symbol} prices are ( bid, ask ) = {ta.get_prices(symbol)}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='order placing function for metatrader 5')

    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-s",
        "--symbol", 
        help="the symbol you need to place the trade on", 
        default=r'',
        choices = ta.tparams.symbols_list,
        required=True
    )
    parser.add_argument(
        "-o"    ,
        "--ordertype"  ,
        help="The order type you need to place",
        choices = ta.tparams.orders_list,
        default=None,
        type=str,
        required=True
    )
    parser.add_argument(
        "-v",
        "--volume",
        help="The number of lots you need to trade",
        default=None,
        type=float,
        required=False
    )

    parser.add_argument(
        "-p",
        "--price",
        help="The price on which you need to place the order",
        default=None,
        type=float,
        required=False
    )

    parser.add_argument(
        "-m",
        "--mode",
        help="The trading style you want to adopt",
        default=ta.tparams.dashboard['defaultTradingMode'],
        type=str,
        choices = ta.tparams.trading_styles
    )

    parser.add_argument(
        "-d",
        "--deltatimeframepair",
        help="Pair containing the period in with candle sticks statistics are calculated in hours and timeframe of the candle sticks",
        default=ta.tparams.dashboard['defaultDeltaTimeframePair'],
        type=str,
        choices = ta.tparams.delta_timeframe_pair_pseudos
    )


    args = parser.parse_args()

    main(
        symbol      =   ta.tparams.symbol_converter(args.symbol),
        ordertype   =   args.ordertype,
        volume      =   args.volume,
        price       =   args.price,
        mode        =   args.mode,
        delta_timeframe_pair = ta.tparams.delta_timeframe_pair_pseudos[args.deltatimeframepair]
    )
