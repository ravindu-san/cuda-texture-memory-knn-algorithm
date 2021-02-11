#include <time.h>
#include <stdlib.h>


void calc_dist_cpu(float *refP, float *queryP, float *dist,int n_ref, int n_query, int n_dim){

    for (int i = 0; i < n_query; i++)
    {
        for(int j = 0; j< n_ref; j++){

            float distance = 0.f;
            
            for(int k = 0; k<n_dim; k++){

                float diff = refP[k + n_dim*n_ref] - queryP[k + n_dim*n_query];
                distance += diff*diff; 
            }

            dist[j + n_ref*i] = distance;
        }
    }
    
}

void sort_cpu(float **dist, int **idx, int n_refP, int n_queryP, int k){

    *idx = (int *)malloc(sizeof(int)*n_refP*n_queryP);

    printf("in sort..\n");
    
    for(int i=0; i<n_queryP; i++){
        for(int j=0; j<n_refP; j++){
            (*idx)[j+i*n_refP] = j;

            // printf("idx init:%d", (*idx)[j+i*n_refP]);
        }
    }

    for(int i=0; i<n_queryP; i++){
       partialQuickSort(*dist, *idx, k, i*n_refP, i*n_refP + n_refP-1);
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

        // printf("\nidx::%d\n", idx[j]);
    }
    
    return j;
}

void partialQuickSort(float *dist, int *idx, int k, int start, int end){

    int pvt = pivot(dist, idx, (start+end)/2, start, end);
    int length = pvt - start + 1;

    // printf("\nlen:%d\n", length);

    if(k < length){
        partialQuickSort(dist, idx, k,start, pvt-1);
    } 
    else if(k>length){
        partialQuickSort(dist, idx, k-length, pvt+1, end);
    }
}



int main()
{
    // cout<<"Hello World\n\n";
    
    int n_refP = 10;
    int n_queryP = 2;
    int k = 6;
    
    srand(time(NULL));
    
    float *distances = (float *)malloc(sizeof(float)*n_refP*n_queryP);
    // int *indexes = (int *)malloc(sizeof(int)*n_refP*n_queryP);
    int *indexes;


    
    for (int i=0; i < n_queryP * n_refP; i++) {
        
        // distances[i] = 10.f*i;
        distances[i] = rand()%1000;
        // indexes[i] = i;
        // indexes[i] = rand()%1000;
    }
    
    ////////////////////////////////////////////////////////////////////////////////
    printf("\n\n before sort.....\n");

     for(int i = 0; i<n_refP; i++){
        
        printf("%f  ", distances[i]);
    }
    
    printf("\n\n");
    
    for(int i = 0; i<n_refP; i++){
        
        printf("%f  ", distances[i+ n_refP]);
    }
    
    printf("\n");
    
    // for(int i = 0; i<n_refP; i++){
        
    //     printf("%d  ", indexes[i]);
    // }
    
    
    ///////////////////////////////////////////////////////////////////////////////
    
    
    // partialQuickSort(distances, indexes, k, 0, n_refP-1);

    sort_cpu(&distances, &indexes, n_refP, n_queryP, k);
    
    
    // vector<int> dist(indexes, indexes+n_refP);
    // partialQuicksort(dist, k, 0, n_refP);

    ///////////////////////////////////////////////////////////////////////////////
    printf("\n\n after sort.....\n");
    
    for(int i = 0; i<n_refP; i++){
        
        printf("%f  ", distances[i]);
    }
    
    printf("\n");
    for(int i = 0; i<n_refP; i++){
        
        printf("%d  ", indexes[i]);
    }
    
    printf("\n");


      for(int i = 0; i<n_refP; i++){
        
        printf("%f  ", distances[i+n_refP]);
    }
    
    printf("\n");
    for(int i = 0; i<n_refP; i++){
        
        printf("%d  ", indexes[i+n_refP]);
    }
    
    printf("\n");
        
    return 0;
}


