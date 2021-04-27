#include <stdio.h>
#include "utilities.h"
// #include <cmath>
// const float infinity = INFINITY;


__global__ void calc_dist_texture(cudaTextureObject_t queryP,
                                  int n_queryP,
                                  float *refP,
                                  int n_refP,
                                  int ref_pitch,
                                  int n_dim,
                                  float *dist,
                                  int dist_pitch)
{
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIndex < n_refP && yIndex < n_queryP)
    {
        float ssd = 0.f;
        for (int i = 0; i < n_dim; i++)
        {
            float tmp = refP[i * ref_pitch + xIndex] - tex2D<float>(queryP, (float)yIndex, (float)i);
            ssd += tmp * tmp;
        }
        
        dist[yIndex * dist_pitch + xIndex] = ssd;

    }else if(yIndex < n_queryP)
    {
        dist[yIndex * dist_pitch + xIndex] = infinity;
    }
    
}




// __global__ void calc_dist_texture(cudaTextureObject_t queryP,
//                                   int n_queryP,
//                                   float *refP,
//                                   int n_refP,
//                                   int ref_pitch,
//                                   int n_dim,
//                                   float *dist)
// {
//     unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
//     unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
//     unsigned int tIdx = threadIdx.x;
//     unsigned int blockDimx = blockDim.x;

//     __shared__ float sharedRef[64];

////    if(threadIdx.y == 0)
//     if(yIndex % blockDim.y == 0){

//         for (size_t i = 0; i < n_dim; i++)
//         {
//             sharedRef[i * blockDimx + tIdx] = refP[i * ref_pitch + xIndex];
//         } 
//     }
//     __syncthreads();

//     if (xIndex < n_refP && yIndex < n_queryP)
//     {
//         float ssd = 0.f;
//         for (int i = 0; i < n_dim; i++)
//         {
//             // float tmp  = tex2D<float>(ref, (float)yIndex, (float)i) - query[i * query_pitch + xIndex];
//             // float tmp = refP[i * ref_pitch + xIndex] - tex2D<float>(queryP, (float)yIndex, (float)i);
//             float tmp = sharedRef[i * blockDimx + tIdx] - tex2D<float>(queryP, (float)yIndex, (float)i);
//             ssd += tmp * tmp;
//         }
//         // dist[yIndex * query_pitch + xIndex] = ssd;
//         dist[yIndex * ref_pitch + xIndex] = ssd;
//     }
// }



__global__ void sort_dist_bitonic(float *distances, int *indexes, int n_refP, int dist_pitch, int n_queryP, const uint stage, const uint passOfStage)
{

    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIndex < n_refP / 2 && yIndex < n_queryP)
    {

        unsigned int pairDistance = 1 << (stage - passOfStage);
        unsigned int blockWidth = 2 * pairDistance;
        unsigned int temp;
        bool compareResult;

        unsigned int leftId = (xIndex & (pairDistance - 1)) + (xIndex >> (stage - passOfStage)) * blockWidth;
        unsigned int rightId = leftId + pairDistance;

        float leftElement, rightElement;
        float greater, lesser;
        int left_idx, right_idx, greater_idx, lesser_idx;


        leftElement = distances[yIndex * dist_pitch + leftId];
        rightElement = distances[yIndex * dist_pitch + rightId];

        if (stage == 0 && passOfStage == 0)
        {
            left_idx = leftId;
            right_idx = rightId;
        }
        else
        {

            left_idx = indexes[yIndex * dist_pitch + leftId];
            right_idx = indexes[yIndex * dist_pitch + rightId];
        }

        
        unsigned int sameDirectionBlockWidth = xIndex >> stage;
        unsigned int sameDirection = sameDirectionBlockWidth & 0x1;

        temp = sameDirection ? rightId : temp;
        rightId = sameDirection ? leftId : rightId;
        leftId = sameDirection ? temp : leftId;

        compareResult = (leftElement < rightElement);
       
        greater = compareResult ? rightElement : leftElement;
        greater_idx = compareResult ? right_idx : left_idx;

        lesser = compareResult ? leftElement : rightElement;
        lesser_idx = compareResult ? left_idx : right_idx;

        distances[yIndex * dist_pitch + leftId] = lesser;
        distances[yIndex * dist_pitch + rightId] = greater;

        indexes[yIndex * dist_pitch + leftId] = lesser_idx;
        indexes[yIndex * dist_pitch + rightId] = greater_idx;
    }
}


//this method is included in utils.h

// unsigned int  getNearestIntOfPow2(int n){

//     if(!(n&(n-1))){//if n is already a power of 2
//         return n;
//     }else
//     {
//         int bitIndex = 0;//equal to log2
//         int shift = 0;
//         // int a[5] = {}

//         bitIndex = (n>0xFFFF) << 4;
//         n >>= bitIndex;

//         shift = (n>0xFF) << 3;
//         n >>= shift;
//         bitIndex |= shift;

//         shift = (n>0xF) << 2;
//         n >>= shift; 
//         bitIndex |= shift;

//         shift = (n>0x3) << 1;
//         n >>= shift; 
//         bitIndex |= shift;

//         bitIndex |= (n >> 1);

//         return 1 << (bitIndex+1);

//     }
    
// }

bool knn_cuda_texture_new(const float *ref_h,
                      int n_refPoints_original,
                      const float *query_h,
                      int n_queryPoints,
                      int n_dimentions,
                      int k,
                      float *dist_h,
                      int *idx_h)
{

    cudaError_t error;
    cudaDeviceProp prop;
    int n_devices;
    int warpSize = 32;

    unsigned int n_refPoints = getNearestIntOfPow2(n_refPoints_original);

    error = cudaGetDeviceCount(&n_devices);
    if (error != cudaSuccess || n_devices == 0)
    {
        printf("ERROR: No CUDA device found\n");
        return false;
    }

    // Select the first CUDA device as default
    error = cudaSetDevice(0);
    if (error != cudaSuccess)
    {
        printf("ERROR: Cannot set the chosen CUDA device\n");
        return false;
    }

    error = cudaGetDeviceProperties(&prop, 0);

    if (error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        return false;
    }

    warpSize = prop.warpSize;

    // Allocate global memory
    float *ref_dev = NULL;
    float *dist_dev = NULL;
    int *idx_dev = NULL;

    size_t ref_pitch_in_bytes;
    size_t dist_pitch_in_bytes;
    size_t idx_pitch_in_bytes;

    error = cudaMallocPitch((void **)&ref_dev, &ref_pitch_in_bytes, n_refPoints_original * sizeof(float), n_dimentions);
    error = cudaMallocPitch((void **)&dist_dev, &dist_pitch_in_bytes, n_refPoints * sizeof(float), n_queryPoints);
    error = cudaMallocPitch((void **)&idx_dev, &idx_pitch_in_bytes, n_refPoints * sizeof(int), n_queryPoints);


    if (error != cudaSuccess)
    {
        printf("Error in cudaMallocPitch: %s\n", cudaGetErrorString(error));
        cudaFree(ref_dev);
        cudaFree(dist_dev);
        cudaFree(idx_dev);

        return false;
    }
    ///////check whether all pitch are equal

    // Deduce pitch value of reference points
    size_t ref_pitch = ref_pitch_in_bytes / sizeof(float);
    size_t dist_pitch = dist_pitch_in_bytes / sizeof(float);
    size_t idx_pitch = idx_pitch_in_bytes / sizeof(int);

    //copy ref data from host to device (in column major)
    error = cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_h, n_refPoints_original * sizeof(float), n_refPoints_original * sizeof(float), n_dimentions, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {

        printf("Error in cudaMemcpy2D: %s\n", cudaGetErrorString(error));
        cudaFree(ref_dev);
        cudaFree(dist_dev);
        cudaFree(idx_dev);

        return false;
    }

    // Allocate CUDA array for query points
    cudaArray *query_array_dev = NULL;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    error = cudaMallocArray(&query_array_dev, &channel_desc, n_queryPoints, n_dimentions);

    if (error != cudaSuccess)
    {

        printf("Error in cudaMallocArray: %s\n", cudaGetErrorString(error));
        cudaFree(ref_dev);
        cudaFree(dist_dev);
        cudaFree(idx_dev);

        return false;
    }

    // Copy query points from host to device
    error = cudaMemcpyToArray(query_array_dev, 0, 0, query_h, n_queryPoints * sizeof(float) * n_dimentions, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {

        printf("Error in cudaMemcpyToArray: %s\n", cudaGetErrorString(error));
        cudaFree(ref_dev);
        cudaFree(dist_dev);
        cudaFree(idx_dev);
        cudaFreeArray(query_array_dev);

        return false;
    }

    // Resource descriptor
    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = query_array_dev;

    // Texture descriptor
    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeClamp;
    tex_desc.addressMode[1] = cudaAddressModeClamp;
    tex_desc.filterMode = cudaFilterModePoint;
    tex_desc.readMode = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    cudaTextureObject_t query_tex_dev = 0;
    error = cudaCreateTextureObject(&query_tex_dev, &res_desc, &tex_desc, NULL);

    if (error != cudaSuccess)
    {

        printf("Error in cudaCreateTextureObject: %s\n", cudaGetErrorString(error));
        cudaFree(ref_dev);
        cudaFree(dist_dev);
        cudaFree(idx_dev);
        cudaFreeArray(query_array_dev);

        return false;
    }

    /////only considered >16
    int block_size_x = warpSize / 2;
    int block_size_y = warpSize / 2;
    int grid_size_x = n_refPoints / block_size_x;
    int grid_size_y = n_queryPoints / block_size_y;

    dim3 block_size = dim3(block_size_x, block_size_y);
    dim3 grid_size = dim3(grid_size_x, grid_size_y);

    calc_dist_texture<<<grid_size, block_size>>>(query_tex_dev, n_queryPoints, ref_dev, n_refPoints_original, ref_pitch, n_dimentions, dist_dev, dist_pitch);

    // cudaDeviceSynchronize();
    cudaThreadSynchronize();

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        printf("Error in calc_dist_texture kernel: %s\n", cudaGetErrorString(error));
        cudaFree(ref_dev);
        cudaFree(dist_dev);
        cudaFree(idx_dev);
        cudaFreeArray(query_array_dev);
        cudaDestroyTextureObject(query_tex_dev);

        return false;
    }

    //////////////////////////////////////////////////////////////////////////////////

    // printf("\n(texture new)before cudaMemcpy2D \n");

    //remove after test.....
    
    // error = cudaMemcpy2D(dist_h, n_refPoints * sizeof(float), dist_dev, dist_pitch_in_bytes, n_refPoints * sizeof(float), n_queryPoints, cudaMemcpyDeviceToHost);
    // error = cudaMemcpy2D(dist_h, n_refPoints_original * sizeof(float), dist_dev, dist_pitch_in_bytes, n_refPoints_original * sizeof(float), n_queryPoints, cudaMemcpyDeviceToHost);

    // printf("\n(texture new)after cudaMemcpy2D \n");

    // if (error != cudaSuccess)
    // {
    //     // printf("error in calc_dist_texture\n");
    //     printf("Error cudaMemcpy2D cudaMemcpyDeviceToHost after calc_dist_texture kernel execution: %s\n", cudaGetErrorString(error));
    //     // exit(-1);
    //     cudaFree(ref_dev);
    //     cudaFree(dist_dev);
    //     cudaFree(idx_dev);
    //     cudaFreeArray(query_array_dev);
    //     cudaDestroyTextureObject(query_tex_dev);

    //     return false;
    // }


    ////////////////////////////////////////////////////////////////////////////////////////

    block_size_x = (n_refPoints / 2) / warpSize;
    block_size_y = n_queryPoints / warpSize;

    block_size = dim3(warpSize, warpSize);
    grid_size = dim3(block_size_x, block_size_y);

    //////////////////////////////////////////////////

     unsigned int numStages = 0, stage = 0, passOfStage = 0, temp = 0;

    for (temp = n_refPoints; temp > 1; temp >>= 1)
    {
        ++numStages;
    }

    for (stage = 0; stage < numStages; ++stage)
    {

        for (passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
        {
            sort_dist_bitonic<<<grid_size, block_size>>>(dist_dev, idx_dev, n_refPoints, dist_pitch, n_queryPoints, stage, passOfStage);
            cudaDeviceSynchronize();
        }
    }

    error = cudaGetLastError();

    if (error != cudaSuccess)

    {
        printf("Error in sort_dist_bitonic kernel: %s\n", cudaGetErrorString(error));
        cudaFree(ref_dev);
        cudaFree(dist_dev);
        cudaFree(idx_dev);
        cudaFreeArray(query_array_dev);
        cudaDestroyTextureObject(query_tex_dev);

        return false;
    }

    error = cudaMemcpy2D(dist_h, k * sizeof(float), dist_dev, dist_pitch_in_bytes, k * sizeof(float), n_queryPoints, cudaMemcpyDeviceToHost);
    error = cudaMemcpy2D(idx_h, k * sizeof(int), idx_dev, idx_pitch_in_bytes, k * sizeof(int), n_queryPoints, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("Error in cudaMemcpy2D cudaMemcpyDeviceToHost after sort_dist_bitonic kernel execution: %s\n", cudaGetErrorString(error));
        cudaFree(ref_dev);
        cudaFree(dist_dev);
        cudaFree(idx_dev);
        cudaFreeArray(query_array_dev);
        cudaDestroyTextureObject(query_tex_dev);

        return false;
    }

     // Memory clean-up
     cudaFree(ref_dev);
     cudaFree(dist_dev);
     cudaFree(idx_dev);
     cudaFreeArray(query_array_dev);
     cudaDestroyTextureObject(query_tex_dev);
 
     return true;

}
