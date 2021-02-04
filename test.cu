#include <time.h>
#include <stdio.h>

#include "knn.cu"
#include "knn-text.cu"
#include "knn-text-old.cu"
#include "utilities.h"



void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    srand(time(NULL));

    // Generate random reference points
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }

    // Generate random query points
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }
}

int main(){

    clock_t knn_glob_start, knn_glob_end, knn_text_new_start, knn_text_new_end, knn_text_old_start, knn_text_old_end;
    double glob_time = 0.0;
    double text_new_time = 0.0;
    double text_old_time = 0.0;

    // int n_refPoints = 8192*2*2*2*2;
    // int n_refPoints = 8192;
    int n_refPoints = 100000;
    int n_queryPoints = 1024;
    int n_dimentions = 4;
    int k = 362;

    // char *refPointsFileName = "testData8192_4.csv";
    // char *queryPointsFileName = "queryPoints_4.csv";

    float *refPoints_h;
    float *refPoints_transpose_h;
    float *queryPoints_h;
    float *queryPoints_transpose_h;
    int *idx_h;
    float *distances_h;

    refPoints_h = (float *)malloc(sizeof(float) * n_dimentions * n_refPoints);
    refPoints_transpose_h = (float *)malloc(sizeof(float) * n_dimentions * n_refPoints);
    queryPoints_h = (float *)malloc(sizeof(float) * n_dimentions * n_queryPoints);
    queryPoints_transpose_h = (float *)malloc(sizeof(float) * n_dimentions * n_queryPoints);
    idx_h = (int *) malloc(sizeof(int) * k * n_queryPoints);
    // distances_h = (float *)malloc(sizeof(float)*n_refPoints*n_queryPoints);
    distances_h = (float *)malloc(sizeof(float)* k *n_queryPoints);

 

   initialize_data(refPoints_h, n_refPoints, queryPoints_h, n_queryPoints, n_dimentions);
    // readRefPoints(refPointsFileName, refPoints_h, n_refPoints, n_queryPoints, n_dimentions);
    // for (int i = 0; i < noOfRefPoints; i++)
    // for (int i = 0; i < 5; i++)
    // {
    //     printf("%d  %f  %f  %f  %f \n", i, refPoints_h[i*n_dimentions + 0], refPoints_h[i*n_dimentions + 1], refPoints_h[i*n_dimentions + 2], refPoints_h[i*n_dimentions + 3]);
    // }
    // readQueryPoints(queryPointsFileName, queryPoints_h, n_dimentions);

   refPoints_transpose_h = transpose(refPoints_h , n_refPoints, n_dimentions);
   queryPoints_transpose_h = transpose(queryPoints_h , n_queryPoints, n_dimentions);

//    for (size_t i = 0; i < n_refPoints* n_dimentions; i++)
//    {
    //    printf("%d    %f\n", i, refPoints_h[i]);
//    }

  
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//knn global

    knn_glob_start = clock();
    knn_cuda_global(refPoints_h, n_refPoints, queryPoints_h, n_queryPoints, n_dimentions, k, distances_h, idx_h);
    knn_glob_end = clock();

    glob_time = (double)(knn_glob_end - knn_glob_start )/CLOCKS_PER_SEC;


    printf("\n\ndistances after sort\n");
    for(int i = 0; i<k ; i++){

        printf("%f  ", distances_h[0 + i]);
    }

    printf("\n\nindexes after sort\n");
    for(int i = 0; i < k ; i++){
        printf("%d  ", idx_h[0 + i]);
    }

    printf("\n\n Global Time:%f\n", glob_time);


    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    //knn texture new

    printf("(texture new)in....\n");

    knn_text_new_start = clock();
    knn_cuda_texture_new(refPoints_transpose_h, n_refPoints, queryPoints_transpose_h, n_queryPoints, n_dimentions, k, distances_h, idx_h);
    knn_text_new_end = clock();

    text_new_time = (double)(knn_text_new_end - knn_text_new_start) / CLOCKS_PER_SEC;


    printf("\n\ndistances after sort\n");
    for(int i = 0; i< k ; i++){

        printf("%f  ", distances_h[0 + i]);
    }

    printf("\n\nindexes after sort\n");
    for(int i = 0; i < k ; i++){
        printf("%d  ", idx_h[0 + i]);
    }

    printf("\n\n Texture New Time:%f\n", text_new_time);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //knn texture old

    knn_text_old_start = clock();
    knn_cuda_texture(refPoints_transpose_h, n_refPoints, queryPoints_transpose_h, n_queryPoints, n_dimentions, k, distances_h, idx_h);
    knn_text_old_end = clock();
    
    text_old_time = (double)(knn_text_old_end - knn_text_old_start) / CLOCKS_PER_SEC;

    printf("\n\ndistances after sort...\n");
    for (int i = 0; i < k; i++)
    {
        printf("%f  ", distances_h[i * n_queryPoints + 0]);
    }

    printf("\n\nindexes after sort...\n");
    for (int i = 0; i < k; i++)
    {
        printf("%d  ", idx_h[i * n_queryPoints + 0]);
    }

    printf("\n\n Texture Old Time:%f\n", text_old_time);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    free(refPoints_h);
    free(refPoints_transpose_h);
    free(queryPoints_h);
    free(queryPoints_transpose_h);
    free(distances_h);//not need if distances are not get back to host
    free(idx_h);

   return 0;



}