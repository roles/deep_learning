#include "Logistic.h"

using namespace std;

Logistic::Logistic(int nIn, int nOut) : MLPLayer(nIn, nOut, Softmax) , TrainComponent(Supervise){ }

void Logistic::trainBatch(int size){
    forward(size);
    backpropagate(size, NULL);
}

void Logistic::runBatch(int size){
    forward(size);
}

void Logistic::setLearningRate(double lr){
    MLPLayer::setLearningRate(lr);
}

void Logistic::setInput(double *input){
    MLPLayer::setInput(input);
}

void Logistic::setLabel(double *label){
    this->label = label;
}

double* Logistic::getOutput(){
    return MLPLayer::getOutput();
}

double* Logistic::getLabel(){
    return label;
}

void Logistic::computeDelta(int size, MLPLayer *prevLayer){
    int numOut = getOutputNumber();
    double *out = getOutput(), *delta = getDelta();

    // get delta dE/d{ai}
    cblas_dcopy(size*numOut, out, 1, delta, 1);
    cblas_daxpy(size*numOut, -1.0, label, 1, delta, 1);
}
