import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) < 2:
    print("Needs an argument file !")
    exit

#print(sys.argv[1][16:-4])

data = pd.read_csv(sys.argv[1], delimiter=';')

plt.figure(figsize=(10, 8), dpi=80)
x = []
y = []
cpu = data['CPU'][0]
gpu = data['GPU'][0]
block_size = data['BLOCK'][0]
linestyles = ['dotted', 'dashed', 'solid']
for i in range(len(data)):
    if data['BLOCK'][i] != block_size or data['CPU'][i] != cpu or data['GPU'][i] != gpu:
        if block_size != 0:
            plt.plot(x, y, label='cpu={} gpu={} bs={}'.format(cpu, gpu, block_size), linestyle=linestyles[2*cpu+gpu-1])
        else:
            plt.plot(x, y, label='CUBLAS')
        cpu = data['CPU'][i]
        gpu = data['GPU'][i]
        block_size = data['BLOCK'][i]
        x = []
        y = []
    x.append(data['K'][i])
    y.append(data['TFLOPS'][i])

plt.plot(x, y, label='cpu={} gpu={} bs={}'.format(cpu, gpu, block_size), linestyle=linestyles[2*cpu+gpu-1])
plt.xlabel('K')
plt.ylabel('TFlops')
plt.legend(loc='upper left')
plt.savefig('figures/figure_{}'.format(sys.argv[1][16:-4]))

