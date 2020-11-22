// #include <stdlib.h>
#include <stdio.h>
#include "utilities.h"

//ToDo
//kernel execution
//copy distance values and check
//remove cls_h and associated read functions

__global__ void calc_dist_texture(cudaTextureObject_t queryP,
                                  int n_queryP,
                                  float *refP,
                                  int n_refP,
                                  int ref_pitch,
                                  int n_dim,
                                  float *dist)
{
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex < n_refP && yIndex < n_queryP)
    {
        float ssd = 0.f;
        for (int i = 0; i < n_dim; i++)
        {
            // float tmp  = tex2D<float>(ref, (float)yIndex, (float)i) - query[i * query_pitch + xIndex];
            float tmp = refP[i * ref_pitch + xIndex] - tex2D<float>(queryP, (float)yIndex, (float)i);
            ssd += tmp * tmp;
        }
        // dist[yIndex * query_pitch + xIndex] = ssd;
        dist[yIndex * ref_pitch + xIndex] = ssd;
    }
}


__global__ void sort_dist_bitonic(float *distances, int *indexes, int n_refP, int dist_pitch, int n_queryP,const uint stage, const uint passOfStage){

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

        // leftElement = distances[yIndex * n_refP + leftId];
        // rightElement =distances[yIndex * n_refP +rightId];
    
        // leftElement_cls = clases[yIndex * n_refP + leftId];
        // rightElement_cls = clases[yIndex * n_refP +rightId];

        leftElement = distances[yIndex * dist_pitch + leftId];
        rightElement =distances[yIndex * dist_pitch +rightId];

        if(stage == 0 && passOfStage == 0){
            left_idx = leftId;
            right_idx = rightId;

        }else{

            left_idx = indexes[yIndex * dist_pitch + leftId];
            right_idx = indexes[yIndex * dist_pitch +rightId];
        }


    
        // leftElement_cls = clases[yIndex * dist_pitch + leftId];
        // rightElement_cls = clases[yIndex * dist_pitch +rightId];

        unsigned int sameDirectionBlockWidth = xIndex >> stage;
        unsigned int sameDirection = sameDirectionBlockWidth & 0x1;
    
        temp = sameDirection ? rightId : temp;
        rightId = sameDirection ? leftId : rightId;
        leftId = sameDirection ? temp : leftId;
    
        compareResult = (leftElement < rightElement);
    
    /////////////////////////////////////////////////////////////////////////////    
        /*add these to a single if else block*/
        greater = compareResult ? rightElement : leftElement;
        // greater_cls = compareResult ? rightElement_cls : leftElement_cls;
        greater_idx = compareResult ? right_idx : left_idx;

        lesser = compareResult ? leftElement : rightElement;
        // lesser_cls = compareResult ? leftElement_cls : rightElement_cls;
        lesser_idx = compareResult ? left_idx : right_idx;
    //////////////////////////////////////////////////////////////////////////////

        distances[yIndex * dist_pitch + leftId] = lesser;
        distances[yIndex * dist_pitch +rightId] = greater;
    
        //dist_pitch = cls_pitch
        // clases[yIndex * dist_pitch + leftId] = lesser_cls;
        // clases[yIndex * dist_pitch +rightId] = greater_cls;
        if(xIndex == 0 && yIndex == 0)
            printf("lesser idx : %d", left_idx);

        indexes[yIndex * dist_pitch + leftId] = lesser_idx;
        indexes[yIndex * dist_pitch +rightId] = greater_idx;
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
    int n_clases = 2;
    int clsOfQuerypts[n_queryPoints];

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


    float *ref_row_maj_h = (float *)malloc(sizeof(float) * n_dimentions * n_refPoints);
    float *ref_h = (float *)malloc(sizeof(float) * n_dimentions * n_refPoints);
    float *dist_h = (float *)malloc(sizeof(float) * n_refPoints * n_queryPoints);
    //remove cls_h
    int *cls_h = (int *)malloc(sizeof(int) * n_refPoints * n_queryPoints);
    int *idx_h = (int *)malloc(sizeof(int) * n_refPoints * n_queryPoints);
    float *query_row_maj_h = (float *)malloc(sizeof(float) * n_dimentions * n_queryPoints);
    float *query_h = (float *)malloc(sizeof(float) * n_dimentions * n_queryPoints);

    char *refPointsFileName = "testData32_4.csv";
    char *queryPointsFileName = "queryPoints1_4.csv";

    readRefPoints(refPointsFileName, ref_row_maj_h, cls_h, n_refPoints, n_queryPoints, n_dimentions);

    ref_h = transpose(ref_row_maj_h, n_refPoints, n_dimentions); //make column major
    free(ref_row_maj_h);

    // for (int i = 0; i < noOfRefPoints; i++)
    for (int i = 0; i < 5; i++)
    {
        printf("%d  %f  %f  %f  %f  %d\n", i, ref_h[i * n_dimentions + 0], ref_h[i * n_dimentions + 1], ref_h[i * n_dimentions + 2], ref_h[i * n_dimentions + 3], cls_h[i]);
    }

    readQueryPoints(queryPointsFileName, query_row_maj_h, n_dimentions);
    query_h = transpose(query_row_maj_h, n_queryPoints, n_dimentions); //make column major
    free(query_row_maj_h);

    

    // Allocate global memory
    float *ref_dev = NULL;
    float *dist_dev = NULL;
    // int *cls_dev = NULL;
    int *idx_dev = NULL;


    size_t ref_pitch_in_bytes;
    size_t dist_pitch_in_bytes;
    // size_t cls_pitch_in_bytes;
    size_t idx_pitch_in_bytes;

    error = cudaMallocPitch((void **)&ref_dev, &ref_pitch_in_bytes, n_refPoints * sizeof(float), n_dimentions);
    error = cudaMallocPitch((void **)&dist_dev, &dist_pitch_in_bytes, n_refPoints * sizeof(float), n_queryPoints);
    // error = cudaMallocPitch((void **)&cls_dev, &cls_pitch_in_bytes, n_refPoints * sizeof(int), n_queryPoints);
    error = cudaMallocPitch((void **)&idx_dev, &idx_pitch_in_bytes, n_refPoints * sizeof(int), n_queryPoints);

    printf("ref_pitch: %d   dist_pitch: %d   idx_pitch: %d\n", ref_pitch_in_bytes, dist_pitch_in_bytes, idx_pitch_in_bytes);

    if (error != cudaSuccess)
    {
        printf("Error in cudaMallocPitch: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // Deduce pitch value of reference points
    size_t ref_pitch = ref_pitch_in_bytes / sizeof(float);
    size_t dist_pitch = dist_pitch_in_bytes / sizeof(float);
    // size_t cls_pitch = cls_pitch_in_bytes / sizeof(int);
    size_t idx_pitch = idx_pitch_in_bytes / sizeof(int);

    //copy ref data from host to device (in column major)
    error = cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, ref_h, n_refPoints * sizeof(float), n_refPoints * sizeof(float), n_dimentions, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {

        printf("Error in cudaMemcpy2D: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // Allocate CUDA array for query points
    cudaArray *query_array_dev = NULL;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    error = cudaMallocArray(&query_array_dev, &channel_desc, n_queryPoints, n_dimentions);

    if (error != cudaSuccess)
    {

        printf("Error in cudaMemcpy2D: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // Copy query points from host to device
    error = cudaMemcpyToArray(query_array_dev, 0, 0, query_h, n_queryPoints * sizeof(float) * n_dimentions, cudaMemcpyHostToDevice);

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

    printf("\ntexture object created...\n");

    
    int block_size_x = warpSize/2;
    int block_size_y = warpSize/2;
    int grid_size_x = n_refPoints /block_size_x;
    int grid_size_y = n_queryPoints/block_size_y;

    // dim3 block_size = dim3(block_size_x, block_size_y);
    // dim3 grid_size = dim3(grid_size_x, grid_size_y);

    dim3 block_size = dim3(16, 2);
    dim3 grid_size = dim3(n_refPoints/16, 1);


    // calc_dist_texture<<<grid_size, block_size>>>(query_tex_dev, n_queryPoints, ref_dev, n_refPoints, ref_pitch, n_dimentions, dist_dev);
    calc_dist_texture<<<grid_size, block_size>>>(query_tex_dev, n_queryPoints, ref_dev, n_refPoints, ref_pitch, n_dimentions, dist_dev);

    cudaDeviceSynchronize();
    // cudaThreadSynchronize();

    error = cudaGetLastError();

    if (error != cudaSuccess)

    {
        printf("error in kernel\n");
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    error = cudaMemcpy2D(dist_h,  n_refPoints * sizeof(float), dist_dev,  dist_pitch_in_bytes,  n_refPoints * sizeof(float), n_queryPoints, cudaMemcpyDeviceToHost);

    for(int i=0; i< n_refPoints;i++){
        printf("%d) %f  ", i, dist_h[n_refPoints * 1 +i]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // sort_dist_bitonic<<<dim3(1),dim3(32,2)>>>(float *distances, int *clases, int n_refP, int dist_pitch, int n_queryP,const uint stage, const uint passOfStage);


    // int block_count_x = (n_refPoints / 2) / warpSize;
    // int block_count_y = n_queryPoints / warpSize;
    

    // block_size = dim3(warpSize, warpSize);
    // block_size = dim3(warpSize, warpSize);
    // grid_size = dim3(block_count_x, block_count_y);

    block_size = dim3(16, 2);
    grid_size = dim3(1, 1);


    unsigned int numStages = 0, stage = 0, passOfStage = 0, temp = 0;

    for (temp = n_refPoints; temp > 1; temp >>= 1)
    {
        ++numStages;
    }

    for (stage = 0; stage < numStages; ++stage)
    {

        for (passOfStage = 0; passOfStage < stage + 1; ++passOfStage)
        {

            // sort_dist_bitonic<<<grid_size, block_size>>>(distances_d, clases_d, n_refPoints, n_queryPoints, stage, passOfStage);
            sort_dist_bitonic<<<grid_size, block_size>>>(dist_dev, idx_dev, n_refPoints, dist_pitch, n_queryPoints, stage, passOfStage);
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

    error = cudaMemcpy2D(dist_h,  n_refPoints * sizeof(float), dist_dev,  dist_pitch_in_bytes,  n_refPoints * sizeof(float), n_queryPoints, cudaMemcpyDeviceToHost);
    error = cudaMemcpy2D(idx_h,  n_refPoints * sizeof(int), idx_dev,  idx_pitch_in_bytes,  n_refPoints * sizeof(int), n_queryPoints, cudaMemcpyDeviceToHost);

    if (error != cudaSuccess)
    {
        printf("error in cudaMemcpy2D\n");
        printf("Error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }


    printf("\n\nafter sort....\n");
    for(int i=0; i< n_refPoints;i++){
        printf("%f  ", dist_h[n_refPoints * 1 +i]);
    }

    printf("\n\nindexes after sort....\n");
    for(int i=0; i< n_refPoints;i++){
        printf("%d  ", idx_h[n_refPoints * 1 +i]);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////

    cudaFree(ref_dev);
    cudaFree(dist_dev);
    cudaFree(idx_dev);
    cudaFreeArray(query_array_dev);
    free(ref_h);
    free(dist_h);
    free(cls_h);

    return 0;
}