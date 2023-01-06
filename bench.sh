#!/bin/bash

python3 ./plotgen.py "cublas" build/cublas -- -t d

python3 ./plotgen.py "gemm B512" build/gemm -- -b 9 -t d
python3 ./plotgen.py "gemm B1024" build/gemm -- -b 10 -t d
python3 ./plotgen.py "gemm B2048" build/gemm -- -b 11 -t d
python3 ./plotgen.py "gemm B4096" build/gemm -- -b 12 -t d

python3 ./plotgen.py "ppgemm" build/ppgemm -- -t d

python3 ./plotgraph.py "plot/req/cublas.csv" "plot/req/gemm B1024.csv" "plot/req/gemm B2048.csv" "plot/req/gemm B4096.csv" "plot/req/ppgemm.csv"
