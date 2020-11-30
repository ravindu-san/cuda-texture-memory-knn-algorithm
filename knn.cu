//
// Created by ravindu on 2020-11-15.
//

//#include <stdlib.h>
#include <stdio.h>
#include "utilities.h"

__global__ void calc_dist_global_mem(float *refP, float *queryP, float *distances, int n_refP, int n_queryP, int n_dim)
{

    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; //ref points vary across x axis of grid
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y; //query points vary across y axis of grid

    if(xIndex < n_refP && yIndex < n_queryP){

        float sqrd_dist;

        for (int i = 0; i < n_dim; i++)
        {

            float diff = refP[xIndex * n_dim + i] - queryP[yIndex * n_dim + i];//ref points & query points are in row major order
            sqrd_dist += diff * diff;
        }

        distances[yIndex * n_refP + xIndex] = sqrd_dist;

        if(xIndex == 0 && yIndex == 1023){
            printf("distance of ref %d q %d : %f\n", xIndex, yIndex,sqrd_dist);
        }

    }
    
}

// __global__ void sort_dist_bitonic(float *distances, int *clases, int n_refP, int n_queryP,const uint stage, const uint passOfStage){
__global__ void sort_dist_bitonic(float *distances, int *indexes, int n_refP, int n_queryP,const uint stage, const uint passOfStage){

    // uint threadId = get_global_id(0);
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
    
    /////////////////////////////////////////////////////////////////////////////    
        /*add these to a single if else block*/
        greater = compareResult ? rightElement : leftElement;
        greater_idx = compareResult ? right_idx : left_idx;

        lesser = compareResult ? leftElement : rightElement;
        lesser_idx = compareResult ? left_idx : right_idx;
    //////////////////////////////////////////////////////////////////////////////

        distances[yIndex * n_refP + leftId] = lesser;
        distances[yIndex * n_refP +rightId] = greater;
    
        indexes[yIndex * n_refP + leftId] = lesser_idx;
        indexes[yIndex * n_refP + rightId] = greater_idx;

    }
   

}



int main()
{

    // int n_refPoints = 8192;
    // int n_queryPoints = 1024;
    int n_refPoints = 32;
    int n_queryPoints = 2;
    int n_dimentions = 4;
    int k = 4;

    float *refPoints_h, *refPoints_d;
\    int *idx_h, *idx_dev;
    float *queryPoints_h, *queryPoints_d;

    float *distances_h, *distances_d;//distances_h not needed..only for test

    cudaError_t error;
    cudaDeviceProp prop;
    int device_count;
    int warpSize = 32;

    error = cudaGetDeviceCount(&device_count);

    if (error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaGetDeviceProperties(&prop, 0);

    if (error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    warpSize = prop.warpSize;

    printf("device count : %d\n", device_count);
    printf("device name : %s\n", prop.name);
    printf("device total global memory(KB) : %d\n", prop.totalGlobalMem / 1024);
    printf("max texture dimension x : %d    y : %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);

    refPoints_h = (float *)malloc(sizeof(float) * n_dimentions * n_refPoints);
    // idx_h = (int *) malloc(sizeof(int) * n_refPoints * n_queryPoints);
    idx_h = (int *) malloc(sizeof(int) * k * n_queryPoints);
    queryPoints_h = (float *)malloc(sizeof(float) * n_dimentions * n_queryPoints);

    distances_h = (float *)malloc(sizeof(float)*n_refPoints*n_queryPoints);

    // char *refPointsFileName = "testData8192_4.csv";
    // char *queryPointsFileName = "queryPoints_4.csv";
     char *refPointsFileName = "testData32_4.csv";
    char *queryPointsFileName = "queryPoints1_4.csv";

    readRefPoints(refPointsFileName, refPoints_h, n_refPoints, n_queryPoints, n_dimentions);

    // for (int i = 0; i < noOfRefPoints; i++)
    for (int i = 0; i < 5; i++)
    {
        printf("%d  %f  %f  %f  %f \n", i, refPoints_h[i*n_dimentions + 0], refPoints_h[i*n_dimentions + 1], refPoints_h[i*n_dimentions + 2], refPoints_h[i*n_dimentions + 3]);
    }

    readQueryPoints(queryPointsFileName, queryPoints_h, n_dimentions);

    // for (int i = 0; i < n_queryPoints; i++)
    // {
    //     printf("%d  %f  %f  %f  %f \n", i, queryPoints_h[i*n_dimentions + 0], queryPoints_h[i * n_dimentions + 1], queryPoints_h[i*n_dimentions + 2], queryPoints_h[i*n_dimentions + 3]);
    // }

    error = cudaMalloc((void **)&refPoints_d, sizeof(float) * n_dimentions * n_refPoints);
    error = cudaMalloc((void **)&queryPoints_d, sizeof(float) * n_dimentions * n_queryPoints);
    error = cudaMalloc((void **)&idx_dev, sizeof(int) * n_refPoints * n_queryPoints);
    error = cudaMalloc((void **)&distances_d, sizeof(float) * n_refPoints * n_queryPoints);

    if (error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(refPoints_d, refPoints_h, sizeof(float) * n_dimentions * n_refPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(queryPoints_d, queryPoints_h, sizeof(float) * n_dimentions * n_queryPoints, cudaMemcpyHostToDevice);
    

    if (error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    int block_dim = warpSize / 2;
    int grid_dim = (n_refPoints / block_dim);

    dim3 block_size = dim3(block_dim, block_dim);
    dim3 grid_size = dim3(grid_dim, grid_dim);

    printf("\nhello before\n");

    calc_dist_global_mem<<<grid_size, block_size>>>(refPoints_d, queryPoints_d, distances_d, n_refPoints, n_queryPoints, n_dimentions);

    cudaDeviceSynchronize();
    // cudaThreadSynchronize();

    error = cudaGetLastError();

    if (error != cudaSuccess)

    {
        printf("error in kernel\n");
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }


    
    printf("after kernel");
    error = cudaMemcpy(distances_h, distances_d, sizeof(float) * n_refPoints * n_queryPoints, cudaMemcpyDeviceToHost);


    if (error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }


    printf("distances before sort\n");
    for(int i = 0; i<n_refPoints ; i++){

        // printf("hello");

        printf("%d)%f  ", i,distances_h[n_refPoints + i]);

    }


    /////////////////////////////////////////////////////////////////////////////////

    ////handle <32 situations

    // int block_count_x = (n_refPoints / 2) / warpSize;
    // int block_count_x = 16;
    // int block_count_y = n_queryPoints / warpSize;
    // int block_count_y = 2;

    // block_size = dim3(warpSize, warpSize);
    // block_size = dim3(warpSize, warpSize);
    // grid_size = dim3(block_count_x, block_count_y);
    block_size = dim3(warpSize/2, 2);
    grid_size = dim3(1, 1);

    // block_size = dim3(8, 2);
    // grid_size = dim3(1, 1);


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
        printf("error in sort kernel\n");
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMemcpy(distances_h, distances_d, sizeof(float) * n_refPoints * n_queryPoints, cudaMemcpyDeviceToHost);
    error = cudaMemcpy(idx_h, idx_dev, sizeof(int) * n_refPoints * n_queryPoints, cudaMemcpyDeviceToHost);
    error = cudaMemcpy2D(idx_h, k * sizeof(int), idx_dev, n_refPoints*sizeof(int), k * sizeof(int), n_queryPoints, cudaMemcpyDeviceToHost);
    
    printf("\n\ndistances after sort\n");
    for(int i = 0; i<n_refPoints ; i++){

        printf("%f  ", distances_h[n_refPoints + i]);

    }

    printf("\n\nindexes after sort\n");
    for(int i = 0; i < k ; i++){
        // printf("%d  ", idx_h[n_refPoints + i]);
        printf("%d  ", idx_h[k + i]);

    }

    cudaFree(refPoints_d);
    cudaFree(queryPoints_d);
    cudaFree(distances_d);
    free(refPoints_h);
    free(queryPoints_h);
    free(distances_h);//not need if distances are not get back to host

    return 0;
}
