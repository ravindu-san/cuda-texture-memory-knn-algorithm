#include <time.h>
#include <stdlib.h>


void calc_dist_cpu(float *refP, float *queryP, float *dist,int n_ref, int n_query, int n_dim){

    for (int i = 0; i < n_query; i++)
    {
        for(int j = 0; j< n_ref; j++){

            float distance = 0.f;
            
            for(int k = 0; k<n_dim; k++){

                float diff = refP[k + n_dim*j] - queryP[k + n_dim*i]; //row major
                distance += diff*diff; 
            }

            dist[j + n_ref*i] = distance;
        }
    }
    
}


int pivot(float *dist, int *idx, int pIdx, int start, int end){

    float pivot = dist[pIdx];
    int i = start, j = end;

    while (i<=j)
    {
        while(dist[i] < pivot && i<=j){i++;};
        while(dist[j] > pivot && i<=j){j--;};

        if(i==j){
            break;
        }
        else if(dist[i] == dist[j]){
            i++;
            continue;
        }


        float tempDist = dist[i];
        int tempIdx = idx[i];

        dist[i] = dist[j];
        idx[i] =idx[j];

        dist[j] = tempDist;
        idx[j] = tempIdx;

    }
    
    return j;
}

void partialQuickSort(float *dist, int *idx, int k, int start, int end){

    int pvt = pivot(dist, idx, (start+end)/2, start, end);
    int length = pvt - start + 1;

    if(k < length){
        partialQuickSort(dist, idx, k,start, pvt-1);
    } 
    else if(k>length){
        partialQuickSort(dist, idx, k-length, pvt+1, end);
    }
}


void sort_cpu(float *dist, int *idx, int n_refP, int n_queryP, int k){
    
    for(int i=0; i<n_queryP; i++){
        for(int j=0; j<n_refP; j++){
    
            idx[j+i*n_refP] = j;
        }
    }

    for(int i=0; i<n_queryP; i++){
        partialQuickSort(dist, idx, k, i*n_refP, i*n_refP + n_refP-1);
    }
    
}



void knn_cpu
    (float *refP,
     float *queryP,
     int n_ref, 
     int n_query, 
     int n_dim, 
     int k, 
     float *dist_k, 
     int *idx_k)
    {

    float *dist = (float *)malloc(sizeof(float)*n_ref*n_query);
    int *idx = (int *)malloc(sizeof(int)*n_ref*n_query);


    calc_dist_cpu(refP, queryP, dist, n_ref, n_query, n_dim);

    sort_cpu(dist, idx, n_ref, n_query, k);


    for(int i = 0; i< n_query; i++){

        // for(int j=0; j<)
        memcpy(dist_k + i*k, dist + i*n_ref, sizeof(float)*k);
        memcpy(idx_k + i*k, idx + i*n_ref, sizeof(int)*k);
    }
    
    free(dist);
    free(idx);

}
