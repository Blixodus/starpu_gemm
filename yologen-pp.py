#!/usr/bin/env python3

import sys
import csv
import subprocess

if len(sys.argv) != 3:
    print('Usage: yologen-pp.py <csv_file> <program>')
    sys.exit()

file = sys.argv[1]
prog = sys.argv[2]

if file.endswith('.csv') == False:
    print('The file must be a csv file')
    sys.exit()

with open(file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['m', 'n', 'k', 'b', 'perf'])

    for i in range(1, 16):
        sum = 0

        for j in range(3):
            print(f"Running {prog} with m={i}, n={i}, k={i} ({j+1}/3)")
            output = subprocess.check_output([prog, '-m', str(i), '-n', str(i), '-k', str(i), 'b', str(i), '-q'])
            decoded = output.decode('utf-8').strip().split(',')

            sum += float(decoded[4])

        writer.writerow([i, i, i, i, sum / 3])

print("Done")
