#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#define N 8192


uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

// Cd = Ad * Bd
__global__ void cuda_basic_gemm(float* Ad, float* Bd, float* Cd){
	float dp = 0;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col >= N || row >= N) {
		return;
	}

	// do dot product of A row * B col
	for (int i = 0; i < N; i++) {
		dp += Ad[row * N + i] * Bd[i * N + col];
	}

	Cd[row * N + col] = dp;
}

int main() {
    float* devA;
    float* devB;
    float* devC;
	float* A;
	float* B;
	float* C;
	float* val;

	// heap allocate for bigger matrices
	A = (float*) malloc(N * N * sizeof(float));
	B = (float*) malloc(N * N * sizeof(float));
	C = (float*) malloc(N * N * sizeof(float));
	val = (float*) malloc(N * N * sizeof(float));

    // Read matmul from numpy for validation
    // Took this from @geohot: https://github.com/geohot/tinygrad/blob/gemm/extra/gemm/gemm.c#L115
    FILE *f = fopen("/tmp/matmul", "rb");
    if (f == NULL) {
        printf("please pregenerate python /tmp/matmul file\n");
        return -1;
    }
    fread(A, 1, sizeof(float)*N*N, f);
    fread(B, 1, sizeof(float)*N*N, f);
    fread(val, 1, sizeof(float)*N*N, f);
    fclose(f);


    cudaMalloc((void**) &devA, N * N * sizeof(float));
    cudaMalloc((void**) &devB, N * N * sizeof(float));
	cudaMalloc((void**) &devC, N * N * sizeof(float));

	// Copy A and B to device memory
	cudaMemcpy(devA, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(devB, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

	// Thread block size of 32x32
	// Create a grid of 32x32 thread blocks
	dim3 dimGrid(ceil(N/32.0), ceil(N/32.0), 1);
	dim3 dimBlock(32, 32, 1);

	uint64_t start = nanos();
    cuda_basic_gemm<<<dimGrid, dimBlock>>>(devA, devB, devC);
    cudaDeviceSynchronize();

	// dumb CPU matmul
	/* for (int x = 0; x<N; x++) { */
	/* 	for (int y = 0; y<N; y++) { */
	/* 		C[x * N + y] = 0; */
	/* 		printf("C[%d]\n", x*N+y); */
	/* 		for (int k = 0; k < N; k++) { */
	/* 			printf("A[%d] * B[%d]\n", x*N+k, k*N+y); */
	/* 			C[x * N + y] += A[x * N + k] * B[k * N + y]; */
	/* 		} */
	/* 	} */
	/* } */
    uint64_t end = nanos();

	printf("GPU\n");
	double gflop = (2.0*N*N*N)*1e-9;
	double s = (end-start)*1e-9;
	printf("%f GFLOP/S -- %.2f ms\n", gflop/s, s*1e3);

	cudaMemcpy(C, devC, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	for (int k = 0; k < N*N; k++) {
		if (fabsf(C[k] - val[k]) > 1e-3) {
			printf("MISMATCH AT %d, %f != %f\n", k, C[k], val[k]);
			return -1;
		}
	}
	printf("match\n");
	

	cudaFree(devA);
	cudaFree(devB);
	cudaFree(devC);

	free(A);
	free(B);
	free(C);
	free(val);
    return 0;
}
