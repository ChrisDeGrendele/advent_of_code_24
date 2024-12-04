// Add CUDA kernel for distance calculation
__global__ void calculateDistance(int* d_list1, int* d_list2, int* d_result, int nRows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nRows) {
        d_result[idx] = abs(d_list1[idx] - d_list2[idx]);
    }
}
