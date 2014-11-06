#include "MLPLayer.h"

static MLPBuffer temp, temp2;

MLPLayer::MLPLayer(int nIn, int nOut, UnitType t) :
    numIn(nIn), numOut(nOut), unitType(t){
    init(); 
}

MLPLayer::MLPLayer(int nIn, int nOut) : 
    numIn(nIn), numOut(nOut), unitType(Sigmoid){
    init();
}

double* MLPLayer::getOutput() { return out; }

void MLPLayer::init(){
    weight = new double[numIn*numOut];
    bias = new double[numOut];
    delta = NULL;
    out = NULL;

    switch(unitType){
        case Softmax:
            //initializeWeightSigmoid(weight, numIn, numOut);
            memset(weight, 0, numIn*numOut*sizeof(double));
            break;
        case Sigmoid:
            initializeWeightSigmoid(weight, numIn, numOut);
            break;
        case Tanh:
            initializeWeightTanh(weight, numIn, numOut);
            break;
    }
    memset(bias, 0, numOut*sizeof(double));
}

MLPLayer::~MLPLayer(){
    delete[] weight;
    delete[] bias;
    delete[] delta;
    delete[] out;
}

void MLPLayer::forward(int size){
    if(out == NULL){
        out = new double[numOut*size];
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, numOut, numIn,
                1, in, numIn, weight, numOut,
                0, out, numOut);

    cblas_dger(CblasRowMajor, size, numOut,
               1.0, I(), 1, bias, 1, out, numOut);

    switch(unitType){
        case Sigmoid:
            for(int i = 0; i < size*numOut; i++){
                out[i] = sigmoid(out[i]);
            }
            break;
        case Tanh:
            for(int i = 0; i < size*numOut; i++){
                out[i] = tanh(out[i]);
            }
            break;
        case Softmax:
            for(int i = 0; i < size; i++){
                softmax(out+i*numOut, numOut);
            }
            break;
    }
}

void MLPLayer::updateWeight(int size){
    // update weight
    
    // L2 normalization
    cblas_dscal(numIn*numOut, 1.0 - 2.0 * L2Reg * learningRate , weight, 1);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numIn, numOut, size,
                -1.0 * learningRate / size, in, numIn,
                delta, numOut,
                1.0, weight, numOut);
}

void MLPLayer::updateBias(int size){
    // update bias
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numOut, 1, size,
                -1.0 * learningRate / size, delta, numOut,
                I(), 1,
                1.0, bias, 1);
}

void MLPLayer::computeDelta(int size, MLPLayer *prevLayer){
    int prevNumOut = prevLayer->getOutputNumber();

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, numOut, prevNumOut,
                1, prevLayer->getDelta(), prevNumOut, prevLayer->getWeight(), prevNumOut,
                0, temp, numOut);

    double *deriv = temp2;
    switch(unitType){
        case Sigmoid:
            for(int i = 0; i < size*numOut; i++){
                deriv[i] = get_sigmoid_derivative(out[i]);
            }
            break;
        case Tanh:
            for(int i = 0; i < size*numOut; i++){
                deriv[i] = get_tanh_derivative(out[i]);
            }
            break;
    }

    cblas_dsbmv(CblasRowMajor, CblasUpper,
                numOut*size, 0, 1.0, temp,
                1, deriv, 1,
                0, delta, 1);
}

void MLPLayer::backpropagate(int size, MLPLayer *prevLayer){
    if(delta == NULL){
        delta = new double[numOut*size];
    }
    computeDelta(size, prevLayer);
    updateWeight(size);
    updateBias(size);
}
