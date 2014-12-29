#include "Dataset.h"
#include "TrainModel.h"
#include "Utility.h"
#include <cstring>

Dataset::Dataset(){
    numFeature = 0;
    numLabel = 0;
    numTrain = numValid = numTest = 0;
    trainingData = validateData = testData = NULL;
    trainingLabel = validateLabel = testLabel = NULL;
}

void Dataset::dumpTrainingData(const char savefile[]){
    dumpData(savefile, numTrain, trainingData, trainingLabel); 
}

void Dataset::dumpData(const char savefile[], int numData, double* data, double *label){
    FILE* fd = fopen(savefile, "wb+");
    if(fd == NULL){
        printf("file doesn't exist : %s\n", savefile);
        exit(1);
    }

    fwrite(&numData, sizeof(int), 1, fd);
    fwrite(&numFeature, sizeof(int), 1, fd);
    if(label != NULL){
        fwrite(&numLabel, sizeof(int), 1, fd);
    }
    fwrite(data, sizeof(double), numData*numFeature, fd);
    if(label != NULL){
        fwrite(label, sizeof(double), numData*numLabel, fd);
    }

    fclose(fd);
}

Dataset::~Dataset(){
    delete[] trainingData;
    delete[] validateData;
    delete[] testData;
    delete[] trainingLabel;
    delete[] validateLabel;
    delete[] testLabel;
}

SubDataset Dataset::getTrainingSet(){
    return SubDataset(numTrain, numFeature, numLabel, trainingData, trainingLabel);
}

SubDataset Dataset::getValidateSet(){
    return SubDataset(numValid, numFeature, numLabel, validateData, validateLabel);
}

SubDataset Dataset::getTestSet(){
    return SubDataset(numTest, numFeature, numLabel, testData, testLabel);
}

int SequentialBatchIterator::getRealBatchSize(){
    if(cur == (size-1)){
        return data->numSample - batchSize * cur;
    }else{
        return batchSize;
    }
}

RandomBatchIterator::RandomBatchIterator(SubDataset *data, int batchSize) :
    BatchIterator(data, batchSize)
{
    size = (data->numSample-1) / batchSize + 1;
    randIndex = vector<int>(size);
}

void RandomBatchIterator::first(){
    cur = 0;
    for(int i = 0; i < size; i++){
        randIndex[i] = i;
    }
    for(int i = 0; i < size; i++){
        int x = random_int(i, size-1);
        int tmp = randIndex[i];
        randIndex[i] = randIndex[x];
        randIndex[x] = tmp;
    }
}

int RandomBatchIterator::getRealBatchSize(){
    if(randIndex[cur] == (size-1)){
        return data->numSample - batchSize * randIndex[cur];
    }else{
        return batchSize;
    }
}

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

MNISTDataset::~MNISTDataset(){ }

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

TransmissionDataset::TransmissionDataset(Dataset& originData, TrainComponent& component){
    numFeature = component.getTransOutputNumber();
    numLabel = originData.getLabelNumber();
    numTrain = originData.getTrainingNumber();
    numValid = originData.getValidateNumber();

    trainingData = new double[numTrain*numFeature];
    validateData = new double[numValid*numFeature];
    if(originData.getLabelNumber() != 0){

        trainingLabel = new double[numTrain*numLabel];
        memcpy(trainingLabel, originData.getTrainingLabel(0), 
               sizeof(double) * numTrain * numLabel);

        validateLabel = new double[numValid*numLabel];
        memcpy(validateLabel, originData.getValidateLabel(0), 
               sizeof(double) * numValid * numLabel);
    }

    int batchSize = 100;

    component.beforeTraining(batchSize);

    // get training data
    
    SubDataset dataset = originData.getTrainingSet();
    BatchIterator *iter = new SequentialBatchIterator(&dataset, batchSize);

    for(iter->first(); !iter->isDone(); iter->next()){
        int theBatchSize = iter->getRealBatchSize();
        int k = iter->CurrentIndex();

        component.setInput(iter->CurrentDataBatch());
        if(component.getTrainType() == Supervise){
            component.setLabel(iter->CurrentLabelBatch());
        }
        component.runBatch(theBatchSize);
        memcpy(trainingData+k*batchSize*numFeature, 
                component.getTransOutput(), theBatchSize*numFeature*sizeof(double));
    }
    delete iter;

    // get validate data

    dataset = originData.getValidateSet();
    iter = new SequentialBatchIterator(&dataset, batchSize);

    for(iter->first(); !iter->isDone(); iter->next()){
        int theBatchSize = iter->getRealBatchSize();
        int k = iter->CurrentIndex();

        component.setInput(iter->CurrentDataBatch());
        if(component.getTrainType() == Supervise){
            component.setLabel(iter->CurrentLabelBatch());
        }
        component.runBatch(theBatchSize);
        memcpy(validateData+k*batchSize*numFeature, 
                component.getTransOutput(), theBatchSize*numFeature*sizeof(double));
    }
    delete iter;
}

TransmissionDataset::~TransmissionDataset(){ }

void SVMDataset::loadData(const char trainDataFile[],
                           const char validDataFile[])
{
    char line[41000];
    char *saveptr, *saveptr2;

    FILE* trainfd = fopen(trainDataFile, "r");
    fscanf(trainfd, "%d", &numTrain);
    printf("number of training sample : %d\n", numTrain);

    fscanf(trainfd, "%d", &numFeature);
    printf("number of feature : %d\n", numFeature);

    fscanf(trainfd, "%d", &numLabel);
    printf("number of label : %d\n", numLabel);
    getc(trainfd);

    trainingData = new double[numTrain*numFeature];
    trainingLabel = new double[numTrain*numLabel];
    memset(trainingData, 0, sizeof(double)*numTrain*numFeature);
    memset(trainingLabel, 0, sizeof(double)*numLabel*numTrain);

    for(int i = 0; i < numTrain; i++){
        fgets(line, 40000, trainfd); 
        char *token = strtok_r(line, " ", &saveptr);

        int label;
        sscanf(token, "%d", &label);
        label = label - 1;
        trainingLabel[i*numLabel+label] = 1.0;

        while((token = strtok_r(NULL, " ", &saveptr)) != NULL){
            int x;
            double val; 

            sscanf(token, "%d:%lf", &x, &val);
            x = x - 1;
            //trainingData[i*numFeature+x] = val;
            trainingData[i*numFeature+x] = val >= 1.0 ? 1.0 : val;

        }
    }
    fclose(trainfd);

    FILE* validfd = fopen(validDataFile, "r");
    fscanf(validfd, "%d", &numValid);
    printf("number of validate sample : %d\n", numValid);

    fscanf(validfd, "%d", &numFeature);
    printf("number of feature : %d\n", numFeature);

    fscanf(validfd, "%d", &numLabel);
    printf("number of label : %d\n", numLabel);
    getc(validfd);

    validateData = new double[numValid*numFeature];
    validateLabel = new double[numValid*numLabel];
    memset(validateData, 0, sizeof(double)*numValid*numFeature);
    memset(validateLabel, 0, sizeof(double)*numLabel*numValid);

    for(int i = 0; i < numValid; i++){
        fgets(line, 40000, validfd); 
        char *token = strtok_r(line, " ", &saveptr);

        int label;
        sscanf(token, "%d", &label);
        label = label - 1;
        validateLabel[i*numLabel+label] = 1.0;

        while((token = strtok_r(NULL, " ", &saveptr)) != NULL){
            int x;
            double val; 

            sscanf(token, "%d:%lf", &x, &val);
            x = x - 1;
            //validateData[i*numFeature+x] = val;
            validateData[i*numFeature+x] = val >= 1.0 ? 1.0 : val;

        }
    }
    fclose(validfd);
}

void TCGADataset::loadData(const char tcgaTrainDataFile[],
                           const char tcgaValidDataFile[])
{
    SVMDataset::loadData(tcgaTrainDataFile, tcgaValidDataFile);
}
