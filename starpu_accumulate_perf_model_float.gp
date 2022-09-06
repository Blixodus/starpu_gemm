#!/usr/bin/gnuplot -persist

set term postscript eps enhanced color
set output "starpu_accumulate_perf_model_float.eps"
set title "Model for codelet accumulate-perf-model-float on rubik"
set xlabel "Total data size"
set ylabel "Time (ms)"

set key top left
set logscale x
set logscale y

set xrange [1 < * < 10**5 : 10**6 < * < 10**9]

plot	0.001 * 0.000302127 * x ** 1.00771 title "Linear Regression cpu0_impl0 (Comb0)",\
	0.001 * 599.048 * x ** 0.0453067 title "Linear Regression cuda0_impl0 (Comb1)",\
	".//starpu_accumulate_perf_model_float_avg.data" using 1:2:3 with errorlines title "Average cuda0-impl0 (Comb1)"
