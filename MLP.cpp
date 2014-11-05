#include "MLP.h"

using namespace std;

MLP::MLP() : TrainComponent(Supervise), numLayer(0) {}

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

void MLP::setLabel(double *label){
    layers[numLayer-1]->setLabel(label);
}

double* MLP::getOutput(){
    return layers[numLayer-1]->getOutput();
}

double* MLP::getLabel(){
    return layers[numLayer-1]->getLabel();
}
