#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tradlearnXGBoost as x
import argparse
from tradparams import delta_timeframe_pair_pseudos, modes, trends, extensions, learning_rate, learning_trend, mode, modelfile_extension

def main(
    timeframe_pseudo, 
    learningrate,
    trend,
    mode,
    extension
):
    x.test_model(timeframe_pseudo, learningrate=learningrate, trend=trend, mode=mode,extension=extension)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the model')


    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-t",
        "--trend",
        help="The trend you need to learn",
        default=learning_trend,
        choices = trends
    )

    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-l",
        "--learningrate",
        help="Learning step",
        default=learning_rate
    )

    # symbol, order_type, volume, price=None, sl=None, tp=None
    parser.add_argument(
        "-m",
        "--mode",
        help="Mode",
        default=mode,
        choices = modes
    )
    # timeframe_pseudo
    parser.add_argument(
        "-e",
        "--extension",
        help="File Extension for model saving",
        default=modelfile_extension,
        choices = extensions
    )

    parser.add_argument(
        "-p",
        "--timeframepseudo",
        help="Timeframe pseudo",
        choices = delta_timeframe_pair_pseudos.keys(),
        required=True
    )

    args = parser.parse_args()

    main(
        timeframe_pseudo=args.timeframepseudo,
        trend=args.trend,
        mode=args.mode,
        learningrate=args.learningrate,
        extension=args.extension
    )
