#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from tradautotools import SpreadTooHighException, RiskTooHighException
from tradautotools import get_attributes, send_order, candle_size, get_equity, get_prices, get_contract_size, symbol_converter
from tradautotools import delta_timeframe_pair_pseudos, dashboard, orders_list, symbols_list
import argparse



def convert_to_boolean(value):
    return value.lower() in ['true', '1', 'yes', 'y']

def main(
        symbol     ,
        ordertype  ,
        volume     ,
        price      ,
        delta_timeframe_pair
    ):

    try:
        order_type, vol, p, sl, tp, typefilling = get_attributes(
            symbol     ,
            ordertype  ,
            volume     ,
            price      ,
            delta_timeframe_pair
        )
        print(f"Account Expendable Equity = {get_equity()}")

        send_order(symbol=symbol, order_type=order_type, volume=vol, price=p, stoploss=sl, takeprofit=tp, typefilling=typefilling)

        print(f"\nAttributes = ({order_type}, {vol}, {p}, {sl}, {tp})")

        print(f"Symbol {symbol} ( max, average ) candle sizes : {candle_size(symbol, delta_timeframe_pair)}")

        # print(f"Symbol {symbol} conversion factor (conversion pair is {get_conv_pair(symbol)}): {get_conversion_factor(symbol)}")

        print(f"Symbol {symbol} lot size is {get_contract_size(symbol)}")

        print(f"Symbol {symbol} prices are ( bid, ask ) = {get_prices(symbol)}")
    except RiskTooHighException as rth:
        print(rth.message)
    except SpreadTooHighException as sth:
        print(sth.message)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='order placing function for metatrader 5')

    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-s",
        "--symbol",
        help="the symbol you need to place the trade on",
        default=r'',
        choices = symbols_list,
        required=True
    )
    parser.add_argument(
        "-o"    ,
        "--ordertype"  ,
        help="The order type you need to place",
        choices = orders_list,
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
        "-d",
        "--deltatimeframepair",
        help="Pair containing the period in with candle sticks statistics are calculated in hours and timeframe of the candle sticks",
        default = dashboard['defaultDeltaTimeframePair'],
        type=str,
        choices = delta_timeframe_pair_pseudos
    )


    args = parser.parse_args()

    main(
        symbol      =   symbol_converter(args.symbol),
        ordertype   =   args.ordertype,
        volume      =   args.volume,
        price       =   args.price,
        delta_timeframe_pair = delta_timeframe_pair_pseudos[args.deltatimeframepair]
    )
