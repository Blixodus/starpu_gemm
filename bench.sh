#!/bin/bash

python3 ./yologen-pp.py "cublas" build/cublas

python3 ./yologen-pp.py "gemm B256" build/gemm -- -b 8
python3 ./yologen-pp.py "gemm B512" build/gemm -- -b 9
python3 ./yologen-pp.py "gemm B1024" build/gemm -- -b 10
python3 ./yologen-pp.py "gemm B2048" build/gemm -- -b 11
python3 ./yologen-pp.py "gemm B4096" build/gemm -- -b 12

python3 ./yologen-pp.py "ppgemm" build/ppgemm
