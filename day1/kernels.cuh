#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void calculateDistance(int* d_list1, int* d_list2, int* d_result, int nRows);

#endif