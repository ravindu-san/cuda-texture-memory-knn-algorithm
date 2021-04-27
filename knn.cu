//#include <stdlib.h>
#include <stdio.h>
#include "utilities.h"
// #include <cmath>
// const float infinity = INFINITY;

__global__ void calc_dist_global_mem(float *refP, float *queryP, float *distances, int n_refP_original, int n_refP, int n_queryP, int n_dim)
{

    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //ref points vary across x axis of grid
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //query points vary across y axis of grid

    if(xIndex < n_refP_original && yIndex < n_queryP){

        float sqrd_dist;

        for (int i = 0; i < n_dim; i++)
        {

            float diff = refP[xIndex * n_dim + i] - queryP[yIndex * n_dim + i];
            sqrd_dist += diff * diff;
        }

        distances[yIndex * n_refP + xIndex] = sqrd_dist;

    }else if(yIndex < n_queryP)
    {
        distances[yIndex * n_refP + xIndex] = infinity;
    }
    
}



__global__ void sort_dist_bitonic(float *distances, int *indexes, int n_refP, int n_queryP,const uint stage, const uint passOfStage){

    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if(xIndex < n_refP/2 && yIndex < n_queryP){

        unsigned int pairDistance = 1 << (stage - passOfStage);
        unsigned int blockWidth = 2 * pairDistance;
        unsigned int temp;
        bool compareResult;
    
        unsigned int leftId = (xIndex & (pairDistance - 1)) + (xIndex >> (stage - passOfStage)) * blockWidth;
        unsigned int rightId = leftId + pairDistance;
    
        float leftElement, rightElement;
        float greater, lesser;
        int left_idx, right_idx, greater_idx, lesser_idx;

        leftElement = distances[yIndex * n_refP + leftId];
        rightElement =distances[yIndex * n_refP +rightId];
    
        if (stage == 0 && passOfStage == 0)
        {
            left_idx = leftId;
            right_idx = rightId;
        }
        else
        {

            left_idx = indexes[yIndex * n_refP + leftId];
            right_idx = indexes[yIndex * n_refP + rightId];
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

        distances[yIndex * n_refP + leftId] = lesser;
        distances[yIndex * n_refP +rightId] = greater;
    
        indexes[yIndex * n_refP + leftId] = lesser_idx;
        indexes[yIndex * n_refP + rightId] = greater_idx;

    }
   

}


bool knn_cuda_global(const float *ref_h,
                    int n_refPoints_original,
                    const float *query_h,
                    int n_queryPoints,
                    int n_dimentions,
                    int k,
                    float *dist_h,
                    int *idx_h){

    
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

    float *refPoints_d;
    int *idx_dev;
    float *queryPoints_d;
    float *distances_d;

    error = cudaMalloc((void **)&refPoints_d, sizeof(float) * n_dimentions * n_refPoints_original);
    error = cudaMalloc((void **)&queryPoints_d, sizeof(float) * n_dimentions * n_queryPoints);
    error = cudaMalloc((void **)&idx_dev, sizeof(int) * n_refPoints * n_queryPoints);
    error = cudaMalloc((void **)&distances_d, sizeof(float) * n_refPoints * n_queryPoints);


    if (error != cudaSuccess)
    {
        printf("(global) Error in cudaMalloc: %s\n", cudaGetErrorString(error));
        cudaFree(refPoints_d);
        cudaFree(queryPoints_d);
        cudaFree(distances_d);
        cudaFree(idx_dev);
    }

    error = cudaMemcpy(refPoints_d, ref_h, sizeof(float) * n_dimentions * n_refPoints_original, cudaMemcpyHostToDevice);
    error = cudaMemcpy(queryPoints_d, query_h, sizeof(float) * n_dimentions * n_queryPoints, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("(global) Error in cudaMemcpy: %s\n", cudaGetErrorString(error));
        cudaFree(refPoints_d);
        cudaFree(queryPoints_d);
        cudaFree(distances_d);
        cudaFree(idx_dev);
    }

    /////only considered >16
    int block_size_x = warpSize / 2;
    int block_size_y = warpSize / 2;
    int grid_size_x = n_refPoints / block_size_x;
    int grid_size_y = n_queryPoints / block_size_y;

    dim3 block_size = dim3(block_size_x, block_size_y);
    dim3 grid_size = dim3(grid_size_x, grid_size_y);


    calc_dist_global_mem<<<grid_size, block_size>>>(refPoints_d, queryPoints_d, distances_d, n_refPoints_original, n_refPoints, n_queryPoints, n_dimentions);

    cudaDeviceSynchronize();
    // cudaThreadSynchronize();

    error = cudaGetLastError();

    if (error != cudaSuccess)

    {
        printf("(global) Error in calc_dist_global_mem: %s\n", cudaGetErrorString(error));
        cudaFree(refPoints_d);
        cudaFree(queryPoints_d);
        cudaFree(distances_d);
        cudaFree(idx_dev);
    }


    grid_size_x = (n_refPoints / 2) / warpSize;
    grid_size_y = n_queryPoints / warpSize;

    block_size = dim3(warpSize, warpSize);
    grid_size = dim3(grid_size_x, grid_size_y);


    unsigned int numStages = 0, stage = 0, passOfStage = 0, temp = 0;

    for (temp = n_refPoints; temp > 1; temp >>= 1)
    {
        ++numStages;
    }

    for (stage = 0; stage < numStages; ++stage)
    {

        for (passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
        {

            sort_dist_bitonic<<<grid_size, block_size>>>(distances_d, idx_dev, n_refPoints, n_queryPoints, stage, passOfStage);
            cudaDeviceSynchronize();
        }
    }

    error = cudaGetLastError();

    if (error != cudaSuccess)

    {
         printf("(global) Error in sort_dist_bitonic kernel: %s\n", cudaGetErrorString(error));
         cudaFree(refPoints_d);
         cudaFree(queryPoints_d);
         cudaFree(distances_d);
         cudaFree(idx_dev);
 
         return false;
    }

    error = cudaMemcpy2D(dist_h, k * sizeof(float), distances_d, n_refPoints*sizeof(float), k * sizeof(float), n_queryPoints, cudaMemcpyDeviceToHost);
    error = cudaMemcpy2D(idx_h, k * sizeof(int), idx_dev, n_refPoints*sizeof(int), k * sizeof(int), n_queryPoints, cudaMemcpyDeviceToHost);


    if (error != cudaSuccess)

    {
         printf("(global) Error in cudaMemcpy or cudaMemcpy2D: %s\n", cudaGetErrorString(error));
         cudaFree(refPoints_d);
         cudaFree(queryPoints_d);
         cudaFree(distances_d);
         cudaFree(idx_dev);
 
         return false;
    }
    
    cudaFree(refPoints_d);
    cudaFree(queryPoints_d);
    cudaFree(distances_d);
    cudaFree(idx_dev);

    return true;


}
