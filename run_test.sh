#!/bin/bash
# Helper script to run Python tests with CUDA libraries
export LD_LIBRARY_PATH=/usr/local/lib/ollama:$LD_LIBRARY_PATH
python test_gpu.py
