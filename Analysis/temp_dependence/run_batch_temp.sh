#!/bin/bash

for temp in $(seq 0.0 0.05 2.0); do
    python3 LebwohlLasher.py 1000 10 $temp 0
done