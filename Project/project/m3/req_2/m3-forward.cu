#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const float *A, const float *B, float *C,
                                        const int Batch, const int Channel, const int Map_out,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) B[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    size_t b, c, m, h_out, w_out, p, q;

    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // dim3 gridDim(ceil(1.0*Map_out/TILE_WIDTH), ceil((1.0*Height_out * Width_out/TILE_WIDTH)), Batch);  
    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // cudaMalloc(device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    // cudaMalloc(device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    // cudaMalloc(device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    int bz = blockIdx.z, tz = threadIdx.z;

    float val = 0;

    h_out = (by * TILE_WIDTH + ty) / Width_out;
    w_out = (by * TILE_WIDTH + ty) % Width_out;
    b = bz * blockDim.z + tz;

    for (int tileId = 0; tileId < (Channel * K * K - 1) / TILE_WIDTH + 1; tileId++) {

        m = tileId * TILE_WIDTH + tx;
        c = m / (K * K);
        p = (m % (K * K)) / K;
        q = (m % (K * K)) % K;

        if ((bx * TILE_WIDTH + tx) < Map_out && tileId * TILE_WIDTH + ty < Channel * K * K) {
            tileA[tx][ty] = A[(size_t) (bx * TILE_WIDTH + tx) * Channel * K * K + tileId * TILE_WIDTH + ty];
        } else {
            tileA[tx][ty] = 0;
        }
        if ((by * TILE_WIDTH + ty) < Height_out * Width_out && tileId * TILE_WIDTH + tx < Channel * K * K && b < Batch) {
            tileB[tx][ty] = in_4d(b, c, h_out + p, w_out + q);
        } else {
            tileB[tx][ty] = 0;
        }
        __syncthreads();

        if ((bx * TILE_WIDTH + tx) < Map_out && (by * TILE_WIDTH + ty) < Height_out * Width_out) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[tx][i] * tileB[i][ty];
            }
        }
        __syncthreads();
    }

    if ((bx * TILE_WIDTH + tx) < Map_out && (by * TILE_WIDTH + ty) < Height_out * Width_out && b < Batch) {
        C[b * Map_out * Height_out * Width_out + (bx * TILE_WIDTH + tx) * Height_out * Width_out + (by * TILE_WIDTH + ty)] = val;
    }

    #undef in_4d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if ((by * TILE_WIDTH + ty) < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) (by * TILE_WIDTH + ty) * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if ((bx * TILE_WIDTH + tx) < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + (bx * TILE_WIDTH + tx)];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if ((by * TILE_WIDTH + ty) < numCRows && (bx * TILE_WIDTH + tx) < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if ((by * TILE_WIDTH + ty) < numCRows && (bx * TILE_WIDTH + tx) < numCColumns) {
        C[(by * TILE_WIDTH + ty) * numCColumns + (bx * TILE_WIDTH + tx)] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    cudaMalloc(device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    dim3 gridDim(ceil(1.0*Map_out/TILE_WIDTH), ceil((1.0*Height_out * Width_out/TILE_WIDTH)), ceil(1.0*Batch/1));  
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    matrix_unrolling_kernel<<<gridDim, blockDim>>>(device_mask, device_input, device_output, Batch, Channel, Map_out, Height, Width, K);
    cudaDeviceSynchronize();

    // TODO: Set the kernel dimensions and call the matmul kernel
    // dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);  
    // dim3 gridDim2(ceil(1.0*Width_unrolled/TILE_WIDTH), ceil(1.0*Map_out/TILE_WIDTH), 1);
    // matrixMultiplyShared<<<gridDim2, blockDim2>>>(device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);
    // cudaDeviceSynchronize();

    // Permute the result of matrix multiplication
    // const int out_image_size = Height_out * Width_out;
    // dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    // matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        // matmul_output, device_output, Map_out, Batch, out_image_size
    // );

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMemcpy(host_output, device_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

