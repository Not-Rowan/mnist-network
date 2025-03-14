#!/bin/bash

# run the program
clear && clear && clang main.c -L/usr/local/lib -lneuralNetworkLib -I/usr/local/include -o main && ./main && python3 graph.py
