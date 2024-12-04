#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <unordered_map>

#include "kernels.cuh"
using namespace std;

int main() {
  int nRows = 0;
  int x;

  ifstream inFile;
    
  inFile.open("input");
  if (!inFile) {
    cout << "Unable to open file" << endl;
    exit(1); // terminate with error
  }
    
  //get number of rows in file
  while (inFile >> x) {
    nRows++;
  }
  inFile.close();

  nRows /= 2; //two ints per column.

  //allocate storage for each row.
  int list1[nRows];
  int list2[nRows];
    
  inFile.open("input");
  if (!inFile) {
    cout << "Unable to open file";
    exit(1); // terminate with error
  }
  int i = 0;
  while (inFile >> list1[i] >> list2[i]) {
    i++;
  }

  inFile.close();


  //sort list1 and list2
  std::sort(list1, list1+nRows);
  std::sort(list2, list2+nRows);

  //allocate memory on GPU/
  int *d_list1, *d_list2, *d_result;
  cudaMalloc(&d_list1, nRows * sizeof(int));
  cudaMalloc(&d_list2, nRows * sizeof(int));
  cudaMalloc(&d_result, nRows * sizeof(int));

  // Copy data to GPU
  cudaMemcpy(d_list1, list1, nRows * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_list2, list2, nRows * sizeof(int), cudaMemcpyHostToDevice);


  // Launch kernel
  int blockSize = 256;
  int numBlocks = (nRows + blockSize - 1) / blockSize;
  calculateDistance<<<numBlocks, blockSize>>>(d_list1, d_list2, d_result, nRows);


  thrust::device_ptr<int> d_result_ptr(d_result);
  int distance = thrust::reduce(d_result_ptr, d_result_ptr + nRows, 0, thrust::plus<int>());

  std::cout << "Distance Between the Lists (Part 1): " << distance << std::endl;


  //compute similarity score, we'll do this on the cpu with a hashmap taking advantage of O(1) lookup time.
  std::unordered_map<int, int> list2_map;
  for (int i = 0; i < nRows; i++) {
    if (list2_map.find(list2[i]) == list2_map.end())
      {
        list2_map[list2[i]] = 1;
      }
    else
      {
        list2_map[list2[i]]++;
      }
  }

  int similarity_score = 0;
  for (int i = 0; i < nRows; i++) {
    similarity_score += list1[i] * list2_map[list1[i]];
  }

  std::cout << "Similarity Score (Part 2): " << similarity_score << std::endl;


  return 0;
}
