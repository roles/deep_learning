#include "TrainComponent.h"
#include <cstring>
#include <cstdlib>
#include <ctime>

IModel::IModel(const char* name) : modelFile(NULL), vabias(0.0)
{
    strcpy(modelName, name);
}

IModel::~IModel(){
    delete modelFile;
}

void IModel::setModelFile(const char *modelFile){
    delete[] this->modelFile;

    this->modelFile = new char[strlen(modelFile)+1];
    strcpy(this->modelFile, modelFile);
}

void IModel::saveModel(){
    FILE *fd;

    if(getModelFile() == NULL) return;
    fd = fopen(getModelFile(), "wb+");

    if(fd == NULL){
        fprintf(stderr, "cannot open file %s\n", getModelFile());
        exit(1);
    }

    saveModel(fd);

    fclose(fd);
}

TrainComponent::TrainComponent(TrainType t, const char* name) 
    : trainType(t), IModel(name) { }

TrainComponent::~TrainComponent(){ }

void TrainComponent::beforeTraining(int){ }

void TrainComponent::afterTraining(int){ 
    if(getModelFile() == NULL){
        char* newModelFile = new char[200];    
        time_t now = time(NULL);
        struct tm tm = *localtime(&now);
        sprintf(newModelFile, "result/%sModel_%d%d%d_%d_%d.dat", 
                getModelName(),
                tm.tm_year + 1900,
                tm.tm_mon + 1,
                tm.tm_mday, 
                tm.tm_hour,
                tm.tm_min);
        setModelFile(newModelFile);
        delete[] newModelFile;
    }
    saveModel();
}

UnsuperviseTrainComponent::UnsuperviseTrainComponent(const char* name)
    : TrainComponent(Unsupervise, name) { }

UnsuperviseTrainComponent::~UnsuperviseTrainComponent(){ }

