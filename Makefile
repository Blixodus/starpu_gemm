CC = nvcc
CFLAGS += $(shell pkg-config --cflags starpu-1.3)
LIBS += $(shell pkg-config --libs starpu-1.3)
CFLAGS += $(shell pkg-config --cflags openblas)
LIBS += $(shell pkg-config --libs openblas)
CFLAGS += 
WARN = -Wall -Wextra -Wpedantic

gemm : gemm.cpp gemm_func.o bzero_func.o accumulate_func.o cublas_perf.o
	$(CC) $(CFLAGS) gemm.cpp gemm_func.o bzero_func.o accumulate_func.o cublas_perf.o -o gemm $(LIBS) -lcublas

gemm_func.o : gemm_func.cu gemm_func.h
	$(CC) $(CFLAGS) -c gemm_func.cu -o gemm_func.o

bzero_func.o : bzero_func.cu bzero_func.h
	$(CC) $(CFLAGS) -c bzero_func.cu -o bzero_func.o

accumulate_func.o : accumulate_func.cu accumulate_func.h
	$(CC) $(CFLAGS) -c accumulate_func.cu -o accumulate_func.o

cublas_perf.o : cublas_perf.cu cublas_perf.h
	$(CC) $(CFLAGS) -c cublas_perf.cu -o cublas_perf.o

clean :
	rm gemm
	rm *.o
