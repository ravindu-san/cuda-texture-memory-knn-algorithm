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

    float sqrd_dist;

    for (int i = 0; i < n_dim; i++)
    {

        float diff = refP[xIndex * n_dim + i] - queryP[yIndex * n_dim + i];//ref points & query points are in row major order
        sqrd_dist += diff * diff;
    }

    distances[yIndex * n_refP + xIndex] = sqrd_dist;

    // if(xIndex == 0 && yIndex == 0){
        printf("distance of ref %d q %d : %f\n", xIndex, yIndex,sqrd_dist);
    // }
}

int main()
{

    int n_refPoints = 8192;
    int n_queryPoints = 2048;
    int n_dimentions = 4;

    float *refPoints_h, *refPoints_d;
    // ClassAndDist *classAndDistArr_h, *classAndDistArr_d;
    int *clases_h, *clases_d;
    float *queryPoints_h, *queryPoints_d;

    float *distances_d;

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
    // classAndDistArr_h = (ClassAndDist *)malloc(sizeof(ClassAndDist) * n_refPoints * n_queryPoints);
    clases_h = (int *) malloc(sizeof(int) * n_refPoints * n_queryPoints);
    queryPoints_h = (float *)malloc(sizeof(float) * n_dimentions * n_queryPoints);

    char *refPointsFileName = "testData8192_4.csv";
    char *queryPointsFileName = "queryPoints_4.csv";

    readRefPoints(refPointsFileName, refPoints_h, clases_h, n_refPoints, n_queryPoints, n_dimentions);

    // for (int i = 0; i < noOfRefPoints; i++)
    // for (int i = 0; i < 5; i++)
    // {
    //     printf("%d  %f  %f  %f  %f  %d\n", i, refPoints_h[i*n_dimentions + 0], refPoints_h[i*n_dimentions + 1], refPoints_h[i*n_dimentions + 2], refPoints_h[i*n_dimentions + 3], classAndDistArr_h[i].cls);
    // }

    readQueryPoints(queryPointsFileName, queryPoints_h, n_dimentions);

    // for (int i = 0; i < n_queryPoints; i++)
    // {
    //     printf("%d  %f  %f  %f  %f \n", i, queryPoints_h[i*n_dimentions + 0], queryPoints_h[i * n_dimentions + 1], queryPoints_h[i*n_dimentions + 2], queryPoints_h[i*n_dimentions + 3]);
    // }


    error = cudaMalloc((void **)&refPoints_d, sizeof(float) * n_dimentions * n_refPoints);
    error = cudaMalloc((void **)&queryPoints_d, sizeof(float) * n_dimentions * n_queryPoints);
    // // error = cudaMalloc((void **)classAndDistArr_d, sizeof(ClassAndDist) * n_refPoints * n_queryPoints);
    error = cudaMalloc((void **)&clases_d, sizeof(int) * n_refPoints * n_queryPoints);
    error = cudaMalloc((void **)&distances_d, sizeof(float) * n_refPoints * n_queryPoints);

    if (error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    cudaMemcpy(refPoints_d, refPoints_h, sizeof(float) * n_dimentions * n_refPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(queryPoints_d, queryPoints_h, sizeof(float) * n_dimentions * n_queryPoints, cudaMemcpyHostToDevice);
    cudaMemcpy(clases_d, clases_h, sizeof(int) * n_refPoints * n_queryPoints, cudaMemcpyHostToDevice);


    int block_dim = warpSize / 2;
    int grid_dim = (n_refPoints / block_dim);

    dim3 block_size = dim3(block_dim, block_dim);
    dim3 grid_size = dim3(grid_dim, grid_dim);

    printf("\nhello before\n");

    calc_dist_global_mem<<<grid_size, block_size>>>(refPoints_d, queryPoints_d, distances_d, n_refPoints, n_queryPoints, n_dimentions);

    cudaDeviceSynchronize();
    // cudaThreadSynchronize();
    
    printf("after kernel");
    // cudaMemcpy(classAndDistArr_h, classAndDistArr_d, sizeof(ClassAndDist) * n_refPoints * n_queryPoints, cudaMemcpyDeviceToHost);

    // for(int i = 0; i<1 ; i++){

    //     printf("hello");

    //     printf("\n%f\n",_h[0].distance);

    // }

    cudaFree(refPoints_d);
    cudaFree(queryPoints_d);
    free(refPoints_h);
    free(queryPoints_h);
    free(clases_h);

    return 0;
}
