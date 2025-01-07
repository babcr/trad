#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import tradlearnXGBoost as x
import argparse
from tradparams import ranges, trends, extensions, learning_rate, learning_trend, mode, modelfile_extension

def main(
    learningrate,
    trend,
    mode,
    extension
):
    x.test_model(learningrate=learningrate, trend=trend, mode=mode,extension=extension)

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
        choices = ranges
    )

    parser.add_argument(
        "-e",
        "--extension",
        help="File Extension for model saving",
        default=modelfile_extension,
        choices = extensions
    )

    args = parser.parse_args()

    main(
        trend=args.trend,
        mode=args.mode,
        learningrate=args.learningrate,
        extension=args.extension
    )
