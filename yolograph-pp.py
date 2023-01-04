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

for file in sys.argv[1:]:
    if file.endswith('.csv') == False:
        print('The file must be a csv file')
        sys.exit()

    with open(file, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    if len(data[0]) != 5:
        print('The csv file must have 4 columns')
        sys.exit()

    if data[0][0] != 'm' or data[0][1] != 'n' or data[0][2] != 'k' or data[0][3] != 'b' or data[0][4] != 'perf':
        print('The csv file must have the columns m, n, k and perf')
        sys.exit()

    x = x if x is not None else [math.log2(int(row[0])) for row in data[1:]]
    y.append([float(row[4]) for row in data[1:]])
    labels.append(Path(file).stem)

for i in range(len(y)):
    plt.plot(x[:len(y[i])], y[i], label=labels[i], marker='o')

plt.xlabel('Matrix size (log)')
plt.ylabel('Performance (gflops)')
plt.legend()

os.makedirs('plot', exist_ok=True)
plt.savefig('plot/' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png')
