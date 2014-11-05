#include "Dataset.h"

Dataset::Dataset(){
    numFeature = 0;
    numTrain = numValid = numTest = 0;
    trainingData = validateData = testData = NULL;
    trainingLabel = validateLabel = testLabel = NULL;
}

Dataset::~Dataset(){}

static int changeEndian(int a){
    int nbyte = sizeof(int);
    char *p = (char*) &a;
    char tmp;
    for(int i = 0, j = nbyte - 1; i < j; i++, j--){
        tmp = p[i]; p[i] = p[j]; p[j] = tmp; 
    }
    return a;
}

void MNISTDataset::loadData(const char *trainingDataFile, const char *trainingLabelFile){

    int magicNum, numImage, numRow, numCol;
    uint8_t pixel, label;

    FILE *traingDataFd = fopen(trainingDataFile, "rb");

    //读入样本数据，50000个做training，10000个做validate
    printf("loading training data...\n");
    fread(&magicNum, sizeof(int), 1, traingDataFd);
    magicNum = changeEndian(magicNum);
    printf("magic number : %d\n", magicNum);

    fread(&numImage, sizeof(int), 1, traingDataFd);
    numImage = changeEndian(numImage);
    printf("number of images : %d\n", numImage);

    fread(&numRow, sizeof(int), 1, traingDataFd);
    numRow = changeEndian(numRow);
    printf("number of rows: %d\n", numRow);

    fread(&numCol, sizeof(int), 1, traingDataFd);
    numCol = changeEndian(numCol);
    printf("number of columns : %d\n", numCol);

    numFeature = numRow * numCol;
    numTrain = 50000;
    numValid = 10000;
    
    trainingData = new double[numTrain*numFeature];
    validateData = new double[numValid*numFeature];

    for(int i = 0; i < numTrain; i++)
        for(int j = 0; j < numFeature; j++){
            fread(&pixel, sizeof(uint8_t), 1, traingDataFd);
            trainingData[numFeature*i+j] = pixel / 255.0;   //除以255.0进行归一化
        }

    for(int i = 0; i < numValid; i++)
        for(int j = 0; j < numFeature; j++){
            fread(&pixel, sizeof(uint8_t), 1, traingDataFd);
            validateData[numFeature*i+j] = pixel / 255.0;
        }

    fclose(traingDataFd);

    printf("loading training label...\n");
    
    FILE *traingLabelFd = fopen(trainingLabelFile, "rb");

    fread(&magicNum, sizeof(int), 1, traingLabelFd);
    magicNum = changeEndian(magicNum);
    printf("magic number : %d\n", magicNum);

    fread(&numImage, sizeof(int), 1, traingLabelFd);
    numImage = changeEndian(numImage);
    printf("number of images : %d\n", numImage);

    numLabel = 10;

    trainingLabel = new double[numTrain*numLabel];
    validateLabel = new double[numValid*numLabel];
    memset(trainingLabel, 0, numTrain*numLabel*sizeof(double));
    memset(validateLabel, 0, numValid*numLabel*sizeof(double));

    for(int i = 0; i < numTrain; i++){
        fread(&label, sizeof(uint8_t), 1, traingLabelFd);
        trainingLabel[i*numLabel+label] = 1.0;
    }

    for(int i = 0; i < numValid; i++){
        fread(&label, sizeof(uint8_t), 1, traingLabelFd);
        validateLabel[i*numLabel+label] = 1.0;
    }

    fclose(traingLabelFd);

    printf("loading ok...\n");
}

MNISTDataset::~MNISTDataset(){
    delete[] trainingData;
    delete[] validateData;
    delete[] testData;
    delete[] trainingLabel;
    delete[] validateLabel;
    delete[] testLabel;
}

void TrivialDataset::loadData(const char *trainingDataFile, const char *trainingLabelFile){

    uint8_t label;

    FILE *traingDataFd = fopen(trainingDataFile, "rb");

    printf("loading training data...\n");

    fread(&numTrain, sizeof(int), 1, traingDataFd);
    printf("number of training sample : %d\n", numTrain);

    fread(&numValid, sizeof(int), 1, traingDataFd);
    printf("number of validate sample : %d\n", numValid);

    fread(&numFeature, sizeof(int), 1, traingDataFd);
    printf("number of feature : %d\n", numFeature);

    trainingData = new double[numTrain*numFeature];
    validateData = new double[numValid*numFeature];

    for(int i = 0; i < numTrain; i++)
        for(int j = 0; j < numFeature; j++){
            fread(&trainingData[numFeature*i+j], sizeof(double), 1, traingDataFd);
        }

    for(int i = 0; i < numValid; i++)
        for(int j = 0; j < numFeature; j++){
            fread(&validateData[numFeature*i+j], sizeof(double), 1, traingDataFd);
        }

    fclose(traingDataFd);

    printf("loading training label...\n");
    
    FILE *traingLabelFd = fopen(trainingLabelFile, "rb");

    fread(&numLabel, sizeof(int), 1, traingLabelFd);
    printf("number of label : %d\n", numLabel);

    trainingLabel = new double[numTrain*numLabel];
    validateLabel = new double[numValid*numLabel];
    memset(trainingLabel, 0, numTrain*numLabel*sizeof(double));
    memset(validateLabel, 0, numValid*numLabel*sizeof(double));

    for(int i = 0; i < numTrain; i++){
        fread(&label, sizeof(uint8_t), 1, traingLabelFd);
        trainingLabel[i*numLabel+label] = 1.0;
    }

    for(int i = 0; i < numValid; i++){
        fread(&label, sizeof(uint8_t), 1, traingLabelFd);
        validateLabel[i*numLabel+label] = 1.0;
    }

    fclose(traingLabelFd);

    printf("loading ok...\n");
}

TrivialDataset::~TrivialDataset(){}
