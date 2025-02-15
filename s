#!/bin/bash

if [ -z "$2" ]; then
    # If the second argument is not provided, use the original command
    python3 "bin\so.py" -s "$1" -o "sell_now"
else
    # If the second argument is provided, include it in the command
    python3 "bin\so.py" -s "$1" -o "sell_now" -d "$2"
fi
