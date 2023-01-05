#!/usr/bin/env python3

import sys
import csv
import subprocess
import os

if len(sys.argv) != 3:
    print('Usage: yologen-pp.py <csv_file> <program>')
    sys.exit()

file = sys.argv[1]
prog = sys.argv[2]
args = sys.argv[3:]

os.makedirs('./plot/req', exist_ok=True)

with open('./plot/req/' + file + ".csv", 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['msize', 'h2d', 'compute', 'd2h', 'perf'])

    for i in range(1, 16):
        h2d = 0
        d2h = 0
        compute = 0
        perf = 0

        for j in range(3):
            print(f"Running {prog} with m={i}, n={i}, k={i} ({j+1}/3)")
            # add the remaning args at the end
            output = subprocess.check_output([prog, '-m', str(i), '-n', str(i), '-k', str(i), 'b', str(i), '-q'] + args)
            decoded = output.decode('utf-8').strip().split(',')

            h2d += float(decoded[1])
            compute += float(decoded[2])
            d2h += float(decoded[3])
            perf += float(decoded[4])

        writer.writerow([i, h2d / 3, compute / 3, d2h / 3, perf / 3])

print("Done")
