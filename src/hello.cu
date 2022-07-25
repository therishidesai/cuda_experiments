#include <stdio.h>
#include <unistd.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<1,1>>>();
	cudaDeviceSynchronize();

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
	printf("SM's: %d", deviceProp.multiProcessorCount);
    return 0;
}
