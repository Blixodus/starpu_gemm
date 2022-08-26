#ifndef CUBLAS_PERF_H
#define CUBLAS_PERF_H

void cublas_perf_test(int m, int n, int k, bool pin, std::ofstream& resultFile);

#endif
