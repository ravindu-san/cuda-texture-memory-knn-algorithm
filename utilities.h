#include <stdlib.h>
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)

typedef struct
{
    int cls;
    float distance;
    
} ClassAndDist;


char *readKernel(char *kernelFileName)
{

    char *source_str = (char *)malloc(MAX_SOURCE_SIZE);
    FILE *fp = fopen(kernelFileName, "r");

    if (!fp)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }

    size_t source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);

    fclose(fp);

    return source_str;
}


void readRefPoints(char *fileName, float *refPoints_h, int noOfRefPoints, int noOfQueryPoints, int noOfAttributes){

    FILE *dataFile = fopen(fileName, "r");

    int index = -1;

        while (!feof(dataFile))
    {
        char line[1024];

        fgets(line, 1024, dataFile);

        if (index >= 0)
        {
            int clsTemp = 0;
            sscanf(line, "%f,%f,%f,%f", &refPoints_h[index*noOfAttributes + 0], &refPoints_h[index*noOfAttributes + 1],&refPoints_h[index*noOfAttributes + 2], &refPoints_h[index*noOfAttributes + 3]);

        }

        index++;
    }

    fclose(dataFile);
}


void readQueryPoints(char *fileName, float *queryPoints_h, int noOfAttributes){

    FILE *dataFile = fopen(fileName, "r");

    int index = -1;

        while (!feof(dataFile))
    {
        char line[1024];

        fgets(line, 1024, dataFile);

        if (index >= 0)
        {

            //only read 4 attributes
            // float *rowOfQueryPoints = (float *)malloc(sizeof(float) * noOfAttributes);

            // sscanf(line, "%f,%f,%f,%d", &rowOfRefPoints[0], &rowOfRefPoints[1], &rowOfRefPoints[2], &classAndDistArr_h[index].class);
            sscanf(line, "%f,%f,%f,%f", &queryPoints_h[index*noOfAttributes + 0], &queryPoints_h[index*noOfAttributes + 1],&queryPoints_h[index*noOfAttributes + 2], &queryPoints_h[index*noOfAttributes + 3]);
            
            // queryPoints_h[index] = rowOfQueryPoints;
        }

        index++;
    }

    fclose(dataFile);
}


int findClassOfQueryPoint(int *sorted_clases, int numberOfClasses, int noOfRefPoints, int queryPointNo, int k)
{
     int classOfQueryPoint = 0;
    int frequenciesOfClasses[numberOfClasses];
    int maxFrequency = 0;

    for (size_t i = 0; i < numberOfClasses; i++)
    {
        frequenciesOfClasses[i] = 0;
    }

    for (size_t i = 0; i < k; i++)
    {
        frequenciesOfClasses[sorted_clases[queryPointNo * noOfRefPoints + i]]++;
    }

    for (size_t i = 0; i < numberOfClasses; i++)
    {
        if (frequenciesOfClasses[i] > maxFrequency)
        {
            maxFrequency = frequenciesOfClasses[i];
            classOfQueryPoint = i;
        }
    }

    return classOfQueryPoint;
}


float *transpose(float* rowMajor1D, int n_rows, int n_cols){

    float *transposeArr = (float *)malloc(sizeof(float) * n_rows * n_cols);

    for (size_t i = 0; i < n_rows; i++)
    {
        for (size_t j = 0; j < n_cols; j++)
        {
            transposeArr[n_rows * j + i] = rowMajor1D[n_cols * i + j]; 
        }
        
    }

    return transposeArr;
    
}


// void initialize_data(float * ref, int     ref_nb, float * query, int     query_nb, int     dim) {

//     // Initialize random number generator
//     srand(time(NULL));

//     // Generate random reference points
//     for (int i=0; i<ref_nb*dim; ++i) {
//         ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
//     }

//     // Generate random query points
//     for (int i=0; i<query_nb*dim; ++i) {
//         query[i] = 10. * (float)(rand() / (double)RAND_MAX);
//     }
// }
