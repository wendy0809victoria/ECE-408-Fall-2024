// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void add(float *input, float *output, int len) {
  int i = 2*blockIdx.x*BLOCK_SIZE + threadIdx.x;
  if (i < len && blockIdx.x >= 1) {
    output[i] += input[blockIdx.x-1];
  }
  if (i+BLOCK_SIZE < len && blockIdx.x >= 1 && i < len) {
    output[i+BLOCK_SIZE] += input[blockIdx.x-1];
  }
}

__global__ void scan(float *input, float *output, float *aux, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float XY[2*BLOCK_SIZE];
  int i = 2*blockIdx.x*BLOCK_SIZE + threadIdx.x;
  if (i < len) {
    XY[threadIdx.x] = input[i];
  } else {
    XY[threadIdx.x] = 0.0;
  }

  if (i+BLOCK_SIZE < len) {
    XY[threadIdx.x+BLOCK_SIZE] = input[i+BLOCK_SIZE];
  } else {
    XY[threadIdx.x+BLOCK_SIZE] = 0.0;
  }

  for (unsigned int stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x+1)*2*stride - 1;
    if (index < 2*BLOCK_SIZE) {
      XY[index] += XY[index-stride];
    }
  }

  for (unsigned int stride = (2*BLOCK_SIZE)/4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x+1)*2*stride - 1;
    if (index+stride < 2*BLOCK_SIZE) {
      XY[index+stride] += XY[index];
    }
  }

  __syncthreads();
  if (i < len) {
    output[i] = XY[threadIdx.x];
  } else {
    // output[i] = 0.0;
  }

  if (i+BLOCK_SIZE < len) {
    output[i+BLOCK_SIZE] = XY[threadIdx.x+BLOCK_SIZE];
  } else {
    // output[i+BLOCK_SIZE] = 0.0;
  }

  __syncthreads();
  if (threadIdx.x == 0 && aux != NULL) {
    aux[blockIdx.x] = XY[2*BLOCK_SIZE-1];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceAux;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceAux, ceil(numElements / (BLOCK_SIZE * 1.0)) * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbCheck(cudaMemset(deviceAux, 0, ceil(numElements / (BLOCK_SIZE * 1.0)) * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid(ceil(numElements / (1.0 * BLOCK_SIZE)), 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, deviceAux, numElements);
  cudaDeviceSynchronize();
  scan<<<dimGrid, dimBlock>>>(deviceAux, deviceAux, NULL, ceil(numElements / (BLOCK_SIZE * 1.0)));
  cudaDeviceSynchronize();
  add<<<dimGrid, dimBlock>>>(deviceAux, deviceOutput, numElements);

  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceAux);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}


