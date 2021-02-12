#include <time.h>
#include <stdio.h>

#include "knn.cu"
#include "knn-text.cu"
#include "knn-text-old.cu"
#include "knn-cpu.c"
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

int main(int argc, char **argv){

     // int n_refPoints = 8192*2*2*2*2;
    // int n_refPoints = 8192;
    int n_refPoints = 100000;
    int n_queryPoints = 1024;
    int n_dimentions = 10;
    int k = 362;
    // int k = 10;

    bool execCpu = 0;
    bool execGlobal = 0;
    bool execTextNew = 0;
    bool execTextOld = 0;


    for(int i = 1; i<argc && argv[i][0] == '-'; i++){

        switch(argv[i][1]){
            case 'r':
                n_refPoints = atoi(argv[++i]);
                break;

            case 'q':
                n_queryPoints = atoi(argv[++i]);
                break;

            case 'd':
                n_dimentions = atoi(argv[++i]);
                break;

            case 'k':
                k = atoi(argv[++i]);
                break;

            case 'c':
                execCpu = atoi(argv[++i]);
                break;

            case 'g':
                execGlobal = atoi(argv[++i]);
                break;

            case 'n':
                execTextNew = atoi(argv[++i]);
                break;

            case 'o':
                execTextOld = atoi(argv[++i]);
                break;

        }
    }

    clock_t knn_glob_start, knn_glob_end, knn_text_new_start, knn_text_new_end, knn_text_old_start, knn_text_old_end, knn_cpu_start, knn_cpu_end;
    double glob_time = 0.0;
    double text_new_time = 0.0;
    double text_old_time = 0.0;
    double cpu_time = 0.0;

    // // int n_refPoints = 8192*2*2*2*2;
    // // int n_refPoints = 8192;
    // int n_refPoints = 100000;
    // int n_queryPoints = 1024;
    // int n_dimentions = 10;
    // int k = 362;
    // // int k = 10;

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


    if(execGlobal){

        printf("\n-----------------Knn Global Memory------------------\n");

        knn_glob_start = clock();
        knn_cuda_global(refPoints_h, n_refPoints, queryPoints_h, n_queryPoints, n_dimentions, k, distances_h, idx_h);
        knn_glob_end = clock();

        glob_time = (double)(knn_glob_end - knn_glob_start )/CLOCKS_PER_SEC;


        printf("\n\ndistances after sort\n");
        for(int i = 0; i<k ; i++){

            // printf("%f  ", distances_h[0 + i]);
            printf("%f  ", distances_h[k + i]);
        }

        printf("\n\nindexes after sort\n");
        for(int i = 0; i < k ; i++){
            // printf("%d  ", idx_h[0 + i]);
            printf("%d  ", idx_h[k + i]);
        }

        printf("\n\n Global Time:%f\n", glob_time);

        printf("\n-----------------------------------------------------\n");
    }

    

    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    //knn texture new

    if(execTextNew){

        printf("\n-----Knn Texture Memory (New Implementation)-----\n");

        knn_text_new_start = clock();
        knn_cuda_texture_new(refPoints_transpose_h, n_refPoints, queryPoints_transpose_h, n_queryPoints, n_dimentions, k, distances_h, idx_h);
        knn_text_new_end = clock();

        text_new_time = (double)(knn_text_new_end - knn_text_new_start) / CLOCKS_PER_SEC;


        printf("\n\ndistances after sort\n");
        for(int i = 0; i< k ; i++){

            // printf("%f  ", distances_h[0 + i]);
            printf("%f  ", distances_h[k + i]);
        }

        printf("\n\nindexes after sort\n");
        for(int i = 0; i < k ; i++){
            // printf("%d  ", idx_h[0 + i]);
            printf("%d  ", idx_h[k + i]);
        }

        printf("\n\n Texture New Time:%f\n", text_new_time);


        printf("\n-----------------------------------------------------\n");


    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //knn texture old

    if(execTextOld){

        printf("\n-----Knn Texture Memory (Old Implementation)-----\n");

        knn_text_old_start = clock();
        bool noErrorInTextOld = knn_cuda_texture(refPoints_transpose_h, n_refPoints, queryPoints_transpose_h, n_queryPoints, n_dimentions, k, distances_h, idx_h);
        knn_text_old_end = clock();

        if(noErrorInTextOld){

            text_old_time = (double)(knn_text_old_end - knn_text_old_start) / CLOCKS_PER_SEC;

            printf("\n\ndistances after sort...\n");
            for (int i = 0; i < k; i++)
            {
                // printf("%f  ", distances_h[i * n_queryPoints + 0]);
                printf("%f  ", distances_h[i * n_queryPoints + 1]);
            }
    
            printf("\n\nindexes after sort...\n");
            for (int i = 0; i < k; i++)
            {
                // printf("%d  ", idx_h[i * n_queryPoints + 0]);
                printf("%d  ", idx_h[i * n_queryPoints + 1]);
            }
    
            printf("\n\n Texture Old Time:%f\n", text_old_time);

        }

       

        printf("\n-----------------------------------------------------\n");

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    //knn-cpu

    if(execCpu){

        printf("\n-------------Knn Serial(CPU) Implementation----------\n");

        knn_cpu_start = clock();
        knn_cpu(refPoints_h, queryPoints_h, n_refPoints, n_queryPoints, n_dimentions, k, distances_h, idx_h);
        knn_cpu_end = clock();

        cpu_time = (double)(knn_cpu_end - knn_cpu_start)/CLOCKS_PER_SEC;


        printf("\n\ndistances after sort\n");
        for(int i = 0; i< k ; i++){

            // printf("%f  ", distances_h[0 + i]);
            printf("%f  ", distances_h[k + i]);
        }

        printf("\n\nindexes after sort\n");
        for(int i = 0; i < k ; i++){
            // printf("%d  ", idx_h[0 + i]);
            printf("%d  ", idx_h[k + i]);
        }

        printf("\n\n CPU Time:%f\n", cpu_time);


        printf("\n-----------------------------------------------------\n");

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    free(refPoints_h);
    free(refPoints_transpose_h);
    free(queryPoints_h);
    free(queryPoints_transpose_h);
    free(distances_h);//not need if distances are not get back to host
    free(idx_h);

   return 0;



}