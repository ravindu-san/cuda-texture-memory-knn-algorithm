//
// Created by ravindu on 2020-11-15.
//

//#include <stdlib.h>
#include <stdio.h>
#include "utilities.h"

__global__ void calc_dist_global_mem(float *refP, float *queryP, int n_refP, int n_queryP, int n_dim){

    

}

int main()
{

    int n_refPoints = 8192;
    int n_queryPoints = 2048;
    int n_dimentions = 4;

    float *refPoints_h;
    ClassAndDist *classAndDistArr_h;
    float *queryPoints_h;

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

    warpSize = prop.warpSize;

    printf("device count : %d\n", device_count);
    printf("device name : %s\n", prop.name);
    printf("device total global memory(KB) : %d\n", prop.totalGlobalMem / 1024);
    printf("max texture dimension x : %d    y : %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);

    refPoints_h = (float *)malloc(sizeof(float) * n_dimentions * n_refPoints);
    classAndDistArr_h = (ClassAndDist *)malloc(sizeof(ClassAndDist) * n_refPoints * n_queryPoints);
    queryPoints_h = (float *)malloc(sizeof(float) * n_dimentions * n_queryPoints);

    char *refPointsFileName = "testData8192_4.csv";
    char *queryPointsFileName = "queryPoints_4.csv";

    readRefPoints(refPointsFileName, refPoints_h, classAndDistArr_h, n_refPoints, n_queryPoints, n_dimentions);

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



    return 0;
}
