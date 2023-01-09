extern "C" void sgemm_(
	char* transA,
	char* transB,
	int* m,
	int* n,
	int* k,
	float* alpha,
	float* A,
	int* lda,
	float* B,
	int* ldb,
	float* beta,
	float* C,
	int* ldc
);

extern "C" void dgemm_(
	char* transA,
	char* transB,
	int* m,
	int* n,
	int* k,
	double* alpha,
	double* A,
	int* lda,
	double* B,
	int* ldb,
	double* beta,
	double* C,
	int* ldc
);

extern "C" float slange_(
    char* norm,
    int* m,
    int* n,
    float* A,
    int* lda,
    float* work
);

extern "C" double dlange_(
    char* norm,
    int* m,
    int* n,
    double* A,
    int* lda,
    double* work
);

extern "C" void sgesvd_(
    char* jobu,
    char* jobvt,
    int* m,
    int* n,
    float* A,
    int* lda,
    float* S,
    float* U,
    int* ldu,
    float* VT,
    int* ldvt,
    float* work,
    int* lwork,
    int* info
);

extern "C" void dgesvd_(
    char* jobu,
    char* jobvt,
    int* m,
    int* n,
    double* A,
    int* lda,
    double* S,
    double* U,
    int* ldu,
    double* VT,
    int* ldvt,
    double* work,
    int* lwork,
    int* info
);
