#!/bin/bash

if [ -z "$2" ]; then
    # If the second argument is not provided, use the original command
    python3 "bin\so.py" -s "$1" -o "buy_stop"
else
    # If the second argument is provided, include it in the command
    python3 "bin\so.py" -s "$1" -o "buy_stop" -d "$2"
fi