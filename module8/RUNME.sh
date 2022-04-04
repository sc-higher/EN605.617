#!/bin/bash
echo "Compiling..."
nvcc assignment.cu -L /usr/local/cuda/lib -lcudart -lcufft -lcublas -lcurand -std=c++11 -o assignment
echo "Done..."
./assignment 1024 256 1024
./assignment 1024 256 10240
./assignment 1024 256 102400