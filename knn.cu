//
// Created by ravindu on 2020-11-15.
//

//#include <stdlib.h>
#include <stdio.h>

int main(){

    cudaError_t error;
    cudaDeviceProp prop;
    int device_count;

    error = cudaGetDeviceCount (&device_count);

    if(error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaGetDeviceProperties(&prop, 0);

    printf("device count : %d\n", device_count);
    printf("device name : %s\n", prop.name);
    printf("device total global memory(KB) : %d\n", prop.totalGlobalMem/1024);
    printf("max texture dimension x : %d    y : %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);

    return 0;
}

