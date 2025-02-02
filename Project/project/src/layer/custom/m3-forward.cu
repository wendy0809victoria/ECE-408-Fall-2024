#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>

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

__global__ void matrix_unrolling_kernel(const half * __restrict__ A, const half * __restrict__ B, half * __restrict__ C,
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

    using namespace nvcuda;

    __shared__ half tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ half tileB[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileC[TILE_WIDTH][TILE_WIDTH];

    // The only dimensions currently supported by WMMA
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);

    // dim3 gridDim(ceil(1.0*Map_out/TILE_WIDTH), ceil((1.0*Height_out * Width_out/TILE_WIDTH)), Batch);  
    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // cudaMalloc(device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    // cudaMalloc(device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    // cudaMalloc(device_output_ptr, Batch * Map_out * Height_out * Width_out * sizeof(float));

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    int bz = blockIdx.z, tz = threadIdx.z;

    // float val = 0;

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

        // Load the inputs
        wmma::load_matrix_sync(a_frag, (half *)tileA, TILE_WIDTH);
        wmma::load_matrix_sync(b_frag, (half *)tileB, TILE_WIDTH);
    
        // Perform the matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();
    }

    wmma::store_matrix_sync((float *)tileC, c_frag, TILE_WIDTH, wmma::mem_row_major);

    if ((bx * TILE_WIDTH + tx) < Map_out && (by * TILE_WIDTH + ty) < Height_out * Width_out && b < Batch) {
        C[b * Map_out * Height_out * Width_out + (bx * TILE_WIDTH + tx) * Height_out * Width_out + (by * TILE_WIDTH + ty)] = tileC[tx][ty];
    }

    #undef in_4d
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
    // const int Height_unrolled = Channel * K * K;
    // const int Width_unrolled = Batch * Height_out * Width_out;

    half *device_input_half;
    cudaMalloc(&device_input_half, Batch * Channel * Height * Width * sizeof(half));
    half *device_mask_half;
    cudaMalloc(&device_mask_half, Map_out * Channel * K * K * sizeof(half));
    half *device_output_half;
    cudaMalloc(&device_output_half, Batch * Map_out * Height_out * Width_out * sizeof(half));

    dim3 gridDim1(ceil((1.0*Batch * Channel * Height * Width/BLOCK_SIZE)), 1, 1);  
    dim3 blockDim1(BLOCK_SIZE, 1, 1);
    floatToHalf<<<gridDim1, blockDim1>>>(device_input_half, device_input, Batch * Channel * Height * Width);

    dim3 gridDim0(ceil((1.0*Map_out * Channel * K * K/BLOCK_SIZE)), 1, 1);  
    dim3 blockDim0(BLOCK_SIZE, 1, 1);
    floatToHalf<<<gridDim0, blockDim0>>>(device_mask_half, device_mask, Map_out * Channel * K * K);

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    dim3 gridDim(ceil(1.0*Map_out/TILE_WIDTH), ceil((1.0*Height_out * Width_out/TILE_WIDTH)), ceil(1.0*Batch/1));  
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    matrix_unrolling_kernel<<<gridDim, blockDim>>>(device_mask_half, device_input_half, device_output_half, Batch, Channel, Map_out, Height, Width, K);
    cudaDeviceSynchronize();

    dim3 gridDim3(ceil((1.0*Batch * Map_out * Height_out * Width_out/BLOCK_SIZE)), 1, 1);  
    dim3 blockDim3(BLOCK_SIZE, 1, 1);
    halfToFloat<<<gridDim3, blockDim3>>>(device_output, device_output_half, Batch * Map_out * Height_out * Width_out);

    cudaFree(device_input_half);
    cudaFree(device_mask_half);
    cudaFree(device_output_half);
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
