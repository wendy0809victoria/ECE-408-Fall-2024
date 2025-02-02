#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_SIZE 3
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[27];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x; 
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int Col = bx * TILE_SIZE + tx;
  int Row = by * TILE_SIZE + ty;
  int Hei = bz * TILE_SIZE + tz;

  int hei_i = Hei - 1;
  int row_i = Row - 1; 
  int col_i = Col - 1; 

  __shared__ float tile[TILE_SIZE+2][TILE_SIZE+2][TILE_SIZE+2];

  float Pvalue = 0.0f;

  if((hei_i >= 0) && (hei_i < z_size) && (row_i >= 0) && (row_i < y_size) && (col_i >= 0) && (col_i < x_size)) {
    tile[tz][ty][tx] = input[hei_i*y_size*x_size + row_i*x_size + col_i];
  } else {
    tile[tz][ty][tx] = 0.0f;
  }

  __syncthreads();

  if (tz < TILE_SIZE && ty < TILE_SIZE && tx <TILE_SIZE) {
    for (int k = 0; k < 3; k++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          Pvalue += deviceKernel[k*TILE_SIZE*TILE_SIZE + i*TILE_SIZE + j] * tile[k+tz][i+ty][j+tx];
        }
      }
    }
    if(Hei < z_size && Row < y_size && Col < x_size) {
      output[Hei*y_size*x_size + Row*x_size + Col] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  for (int dev = 0; dev < deviceCount; dev++) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
  }


  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMalloc((void **) &deviceInput, (inputLength-3) * sizeof(float));
  cudaMemcpy(deviceInput, hostInput+3, (inputLength-3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, 27 * sizeof(float));
  cudaMalloc((void **) &deviceOutput, (inputLength - 3) * sizeof(float));

  //@@ Initialize grid and block dimensions here
  // dim3 dimGrid((x_size + TILE_SIZE - 1) / TILE_SIZE, (y_size + TILE_SIZE - 1) / TILE_SIZE, (z_size + TILE_SIZE - 1) / TILE_SIZE);
  // dim3 dimBlock(TILE_SIZE, TILE_SIZE, TILE_SIZE);
  dim3 dimBlock(TILE_SIZE+2, TILE_SIZE+2, TILE_SIZE+2);
  dim3 dimGrid(ceil(x_size/(1.0*TILE_SIZE)), ceil(y_size/(1.0*TILE_SIZE)), ceil(z_size/(1.0*TILE_SIZE)));
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput+3, deviceOutput, (inputLength-3) * sizeof(float), cudaMemcpyDeviceToHost);



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostKernel);
  free(hostOutput);
  return 0;
}

