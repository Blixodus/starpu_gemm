CC = nvcc
CFLAGS += $(shell pkg-config --cflags starpu-1.3)
LIBS += $(shell pkg-config --libs starpu-1.3)
CFLAGS += $(shell pkg-config --cflags openblas)
LIBS += $(shell pkg-config --libs openblas)
WARN = -Wall -Wextra -Wpedantic

gemm : gemm.cpp blas.o gemm_func.o bzero_func.o accumulate_func.o cublas_perf.o fill_func.o
	$(CC) $(CFLAGS) gemm.cpp blas.o gemm_func.o bzero_func.o accumulate_func.o cublas_perf.o fill_func.o -o gemm $(LIBS) -lcublas

gemm_func.o : gemm_func.cu gemm_func.hpp
	$(CC) $(CFLAGS) -c gemm_func.cu -o gemm_func.o

bzero_func.o : bzero_func.cu bzero_func.hpp
	$(CC) $(CFLAGS) -c bzero_func.cu -o bzero_func.o

accumulate_func.o : accumulate_func.cu accumulate_func.hpp
	$(CC) $(CFLAGS) -c accumulate_func.cu -o accumulate_func.o

fill_func.o : fill_func.cu fill_func.hpp
	$(CC) $(CFLAGS) -c fill_func.cu -o fill_func.o

blas.o : blas.cu blas.hpp
	$(CC) $(CFLAGS) -c blas.cu -o blas.o

cublas_perf.o : cublas_perf.cu cublas_perf.hpp
	$(CC) $(CFLAGS) -c cublas_perf.cu -o cublas_perf.o

clean :
	rm *.o
	rm gemm
