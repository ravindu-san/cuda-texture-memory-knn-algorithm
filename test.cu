#include <time.h>
#include <stdio.h>

#include "knn.cu"
// #include "knn-text.cu"
// #include "knn-text-old.cu"


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

    clock_t knn_glob_start, knn_glob_end;
    double glob_time = 0.0;

    int n_refPoints = 8192*2*2*2*2;
    int n_queryPoints = 1024;
    int n_dimentions = 4;
    int k = 4;

    float *refPoints_h;
    float *queryPoints_h;
    int *idx_h;
    float *distances_h;

    refPoints_h = (float *)malloc(sizeof(float) * n_dimentions * n_refPoints);
    queryPoints_h = (float *)malloc(sizeof(float) * n_dimentions * n_queryPoints);
    idx_h = (int *) malloc(sizeof(int) * k * n_queryPoints);
    distances_h = (float *)malloc(sizeof(float)*n_refPoints*n_queryPoints);

   initialize_data(refPoints_h, n_refPoints, queryPoints_h, n_queryPoints, n_dimentions);

//    for (size_t i = 0; i < n_refPoints* n_dimentions; i++)
//    {
    //    printf("%d    %f\n", i, refPoints_h[i]);
//    }

    knn_glob_start = clock();
    knn_cuda_global(refPoints_h, n_refPoints, queryPoints_h, n_queryPoints, n_dimentions, k, distances_h, idx_h);
    knn_glob_start = clock();

    glob_time = (double)(knn_glob_start - knn_glob_end)/CLOCKS_PER_SEC;


    printf("\n\ndistances after sort\n");
    for(int i = 0; i<n_refPoints ; i++){

        printf("%f  ", distances_h[0 + i]);
    }

    printf("\n\nindexes after sort\n");
    for(int i = 0; i < k ; i++){
        printf("%d  ", idx_h[0 + i]);
    }

    printf("\n\n Global Time:%f\n", glob_time);
   

    free(refPoints_h);
    free(queryPoints_h);
    free(distances_h);//not need if distances are not get back to host
    free(idx_h);

   return 0;



}