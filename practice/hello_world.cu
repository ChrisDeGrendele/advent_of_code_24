#include <stdio.h>

// Kernel function to run on the GPU
__global__ void helloWorld() {
  printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

int main() {
  // Launch the kernel with 1 block of 10 threads
  helloWorld<<<1, 10>>>();

  // Wait for the GPU to finish before accessing results
  cudaDeviceSynchronize();

  return 0;
}
