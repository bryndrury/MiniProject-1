#!/bin/bash

for iter in $(seq 20 20 1000); do
    python3 LebwohlLasher.py $iter 20 0.5 0
done