#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void floatToHalf(__half *dst, const float *src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]); 
    }
}

__global__ void halfToFloat(float *dst, const __half *src, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
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

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]

    // TODO: Insert your input matrix unrolling kernel code here
    size_t b;
    size_t c, s, h_out, w_out, h_unroll, w_unroll, w_base, p, q;
    size_t t = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    size_t W_unroll = Height_out * Width_out;
    size_t num = Channel * W_unroll;

    if (t < num) {
        b = blockIdx.y;
        c = t / W_unroll;
        s = t % W_unroll;
        h_out = s / Width_out;
        w_out = s % Width_out;
        // h_unroll = h_out * Width_out + w_out;
        w_base = c * K * K;
        for (p = 0; p < K; p++) {
            for (q = 0; q < K; q++) {
                w_unroll = h_out * Width_out + w_out;
                h_unroll = w_base + p * K + q;
                size_t index = h_unroll * (Batch*W_unroll) + b * W_unroll + w_unroll;
                output[index] = in_4d(b, c, h_out + p, w_out + q);
                // Batch * Channel * K * K * Height_out * Width_out
                // X_unroll(h_unroll, w_unroll) = in_4d(b, c, h_out + p, w_out + q);
            }
        }
    }

    #undef in_4d
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, __half *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ __half tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ __half tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    __half val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = __float2half(A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx]);
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const __half *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    __half2float(input[m * Batch * image_size + b * image_size + x]);
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
    // float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    // cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));

    // __half *device_input_half;  // Pointer to device memory for storing the unrolled matrix
    // __half *device_mask_half;    // Pointer to device memory for storing the result of matrix multiplication
    // __half *device_output_half;
    // __half *unrolled_matrix;
    __half *matmul_output;
    // cudaMalloc((void**)&device_input_half, (size_t) Batch * Channel * Height * Width * sizeof(__half));
    // cudaMalloc((void**)&unrolled_matrix, (size_t) (Batch * Channel * K * K * Height_out * Width_out) * sizeof(__half));
    cudaMalloc((void**)&matmul_output, (size_t) (Batch * Map_out * Height_out * Width_out) * sizeof(__half));

    // dim3 gridDim0(ceil((1.0*Batch * Channel * Height * Width/BLOCK_SIZE)), 1, 1);  
    // dim3 blockDim0(BLOCK_SIZE, 1, 1);
    // floatToHalf<<<gridDim0, blockDim0>>>(device_input_half, device_input, (size_t) Batch * Channel * Height * Width);
    // cudaDeviceSynchronize();

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    dim3 gridDim(ceil((1.0*Channel*Height_out*Width_out/BLOCK_SIZE)), Batch, 1);  
    dim3 blockDim(BLOCK_SIZE, 1, 1);
    matrix_unrolling_kernel<<<gridDim, blockDim>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);
    cudaDeviceSynchronize();

    // cudaFree(device_input_half);

    // cudaMalloc((void**)&device_mask_half, (size_t) (Map_out * Channel * K * K) * sizeof(__half));
    // cudaMalloc((void**)&matmul_output, (size_t) (Batch * Map_out * Height_out * Width_out) * sizeof(__half));

    // dim3 gridDim1(ceil((1.0*Map_out * Channel * K * K/BLOCK_SIZE)), 1, 1);  
    // dim3 blockDim1(BLOCK_SIZE, 1, 1);
    // floatToHalf<<<gridDim1, blockDim1>>>(device_mask_half, device_mask, Map_out * Channel * K * K);
    // cudaDeviceSynchronize();

    // TODO: Set the kernel dimensions and call the matmul kernel
    dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);  
    dim3 gridDim2(ceil(1.0*Width_unrolled/TILE_WIDTH), ceil(1.0*Map_out/TILE_WIDTH), 1);
    matrixMultiplyShared<<<gridDim2, blockDim2>>>(device_mask, unrolled_matrix, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);
    cudaDeviceSynchronize();

    // cudaFree(unrolled_matrix);
    // cudaFree(device_mask_half);

    // cudaMalloc((void**)&device_output_half, (size_t) (Batch * Map_out * Height_out * Width_out) * sizeof(__half));

    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );

    // dim3 gridDim3(ceil((1.0*Batch * Map_out * Height_out * Width_out/BLOCK_SIZE)), 1, 1);  
    // dim3 blockDim3(BLOCK_SIZE, 1, 1);
    // halfToFloat<<<gridDim3, blockDim3>>>(device_output, device_output_half, (size_t) Batch * Map_out * Height_out * Width_out);
    // cudaDeviceSynchronize();

    cudaFree(unrolled_matrix);
    cudaFree(matmul_output);
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

