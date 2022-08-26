#!/bin/bash

SCHEDS=('dmda')
for sched in ${SCHEDS[*]}
do
    export STARPU_SCHED=$sched
    for var in {14..15}
    do
        make && ./gemm $var 8 14 11 13 | xargs python plot.py
    done
done
