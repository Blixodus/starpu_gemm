#!/usr/bin/env python3

import sys
import csv
import matplotlib.pyplot as plt
import datetime
import os
import math
from pathlib import Path

x = None
y = []
labels = []

os.makedirs('plot', exist_ok=True)

for file in sys.argv[1:]:
    if file.endswith('.csv') == False:
        print('The file must be a csv file')
        sys.exit()

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    if len(data[0]) != 5:
        print('The csv file must have 5 columns')
        sys.exit()

    x = x if x is not None and len(x) > len(data) - 1 else [int(row[0]) for row in data[1:]]
    y.append([float(row[4]) for row in data[1:]])
    labels.append(Path(file).stem)

for i in range(len(y)):
    plt.semilogy(x[:len(y[i])], y[i], label=labels[i], marker='o')

# plt.figure(0)
plt.xlabel('Matrix size (log)')
plt.ylabel('Performance (gflops)')
plt.legend()
plt.savefig('plot/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png')

# plt.figure(1)

