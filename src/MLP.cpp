#include "MLP.h"
#include <typeinfo>

using namespace std;

MLP::MLP() : TrainComponent(Supervise, "MLP"), numLayer(0) {}

MLP::MLP(const char *file) : TrainComponent(Supervise, "MLP"), numLayer(0){
    FILE* fd = fopen(file, "rb");
    loadModel(fd);
    fclose(fd);
}

MLP::~MLP(){
    for(int i = 0; i < numLayer; i++){
        delete layers[i];
    }
}

void MLP::trainBatch(int size){
    for(int i = 0; i < numLayer; i++){
        if(i != 0){
            layers[i]->setInput(layers[i-1]->getOutput());
        }
        layers[i]->forward(size);
    }

    for(int i = numLayer-1; i >= 0; i--){
        if(i == numLayer-1){
            layers[i]->backpropagate(size, NULL);
        }else{
            layers[i]->backpropagate(size, layers[i+1]);
        }
    }
}

void MLP::runBatch(int size){

    for(int i = 0; i < numLayer; i++){
        if(i != 0){
            layers[i]->setInput(layers[i-1]->getOutput());
        }
        layers[i]->forward(size);
    }
}

void MLP::setLearningRate(double lr){
    for(int i = 0; i < numLayer; i++){
        layers[i]->setLearningRate(lr);
    }
}

void MLP::setInput(double *input){
    layers[0]->setInput(input);
}

int MLP::getOutputNumber(){
    return layers[numLayer-1]->getOutputNumber();
}

void MLP::setLabel(double *label){
    layers[numLayer-1]->setLabel(label);
}

double* MLP::getOutput(){
    return layers[numLayer-1]->getOutput();
}

double* MLP::getLabel(){
    return layers[numLayer-1]->getLabel();
}

void MLP::saveModel(FILE *fd){
    fwrite(&numLayer, sizeof(int), 1, fd);
    char layerName[20];
    for(int i = 0; i < numLayer; i++){
        strcpy(layerName, layers[i]->getLayerName());
        fwrite(layerName, sizeof(layerName), 1, fd);
        layers[i]->saveModel(fd);
    }
}

void MLP::loadModel(FILE *fd){
    fread(&numLayer, sizeof(int), 1, fd);
    char layerName[20];

    for(int i = 0; i < numLayer; i++){
        fread(layerName, sizeof(layerName), 1, fd);
        printf("Layer %d : %s\n", (i+1), layerName);
        if(strcmp(layerName, "Tanh") == 0){
            layers[i] = new TanhLayer(fd);
        }else if(strcmp(layerName, "Sigmoid") == 0){
            layers[i] = new SigmoidLayer(fd);
        }else if(strcmp(layerName, "Softmax") == 0){
            layers[i] = new SoftmaxLayer(fd);
        }else if(strcmp(layerName, "Logistic") == 0){
            layers[i] = new Logistic(fd);
        }
    }
}
