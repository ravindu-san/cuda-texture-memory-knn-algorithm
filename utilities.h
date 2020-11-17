#include <stdlib.h>

#define MAX_SOURCE_SIZE (0x100000)


// struct ClassAndDist
// {
//     float distance;
//     int class;
// };
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

void readRefPoints(char *fileName, float *refPoints_h, int *clases_h, int noOfRefPoints, int noOfQueryPoints, int noOfAttributes){

    FILE *dataFile = fopen(fileName, "r");

    int index = -1;

        while (!feof(dataFile))
    {
        char line[1024];

        fgets(line, 1024, dataFile);

        if (index >= 0)
        {

            //only read 4 attributes + class
            // float *rowOfRefPoints = (float *)malloc(sizeof(float) * noOfAttributes);

            // sscanf(line, "%f,%f,%f,%d", &rowOfRefPoints[0], &rowOfRefPoints[1], &rowOfRefPoints[2], &classAndDistArr_h[index].class);

            int clsTemp = 0;
            sscanf(line, "%f,%f,%f,%f,%d", &refPoints_h[index*noOfAttributes + 0], &refPoints_h[index*noOfAttributes + 1],&refPoints_h[index*noOfAttributes + 2], &refPoints_h[index*noOfAttributes + 3], &clsTemp);

            for (int i = 0; i < noOfQueryPoints; i++)
            {
                clases_h[i* noOfRefPoints + index] = clsTemp;
            }
            
            // refPoints_h[index] = rowOfRefPoints;
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