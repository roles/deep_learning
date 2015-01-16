#include "DeepAutoEncoder.h"
#include <cstring>
#include "Config.h"
#include "mkl_cblas.h"
#include "Utility.h"

static double temp[maxUnit*maxUnit];
static double temp2[maxUnit*maxUnit];

EncoderLayer::EncoderLayer(int numIn, int numOut) :
    numIn(numIn), numOut(numOut), y(NULL), h(NULL),
    dy(NULL), dh(NULL), binIn(true), binOut(true)
{
    w = new double[numIn*numOut];
    dw = new double[numIn*numOut];
    b = new double[numOut];
    c = new double[numIn];
    memset(b, 0, numOut*sizeof(double));
    memset(c, 0, numIn*sizeof(double));
    initializeWeightSigmoid(w, numIn, numOut);
}

EncoderLayer::EncoderLayer(int numIn, int numOut, double *w, double *b, double *c) :
    numIn(numIn), numOut(numOut), y(NULL), h(NULL),
    dy(NULL), dh(NULL), binIn(true), binOut(true)
{
    this->w = new double[numIn*numOut];
    this->dw = new double[numIn*numOut];
    this->b = new double[numOut];
    this->c = new double[numIn];
    cblas_dcopy(numIn*numOut, w, 1, this->w, 1);
    cblas_dcopy(numIn, c, 1, this->c, 1);
    cblas_dcopy(numOut, b, 1, this->b, 1);
}

EncoderLayer::~EncoderLayer(){
    delete[] w;
    delete[] dw;
    delete[] b;
    delete[] c;
    delete[] y;
    delete[] h;
    delete[] dy;
    delete[] dh;
}

void EncoderLayer::allocate(int size){
    if(y == NULL) y = new double[size*numIn];
    if(h == NULL) h = new double[size*numOut];
    if(dy == NULL) dy = new double[size*numIn];
    if(dh == NULL) dh = new double[size*numOut];
}

void EncoderLayer::getHFromX(double *x, double *h, int size){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, numOut, numIn,
                1, x, numIn, w, numOut,
                0, h, numOut);

    cblas_dger(CblasRowMajor, size, numOut,
               1.0, I(), 1, b, 1, h, numOut);

    if(binOut){
        for(int i = 0; i < size * numOut; i++){
            h[i] = sigmoid(h[i]);
        }
    }
}

void EncoderLayer::getYFromH(double *h, double *y, int size){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, numIn, numOut,
                1, h, numOut, w, numOut,
                0, y, numIn);

    cblas_dger(CblasRowMajor, size, numIn,
               1.0, I(), 1, c, 1, y, numIn);

    if(binIn){
        for(int i = 0; i < size * numIn; i++){
            y[i] = sigmoid(y[i]);
        }
    }
}

void EncoderLayer::getYDeriv(EncoderLayer* prev, int size){
    if(prev == NULL){
        if(binIn){
            for(int i = 0; i < size * numIn; i++){
                dy[i] = y[i] - x[i];
            }
        }else{
            for(int i = 0; i < size * numIn; i++){
                dy[i] = (y[i] - x[i]) / ((1.0 - y[i] + 1e-10) * (y[i] + 1e-10));
            }
        }
    }else{
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, prev->numOut, prev->numIn,
                    1, prev->dy, prev->numIn, prev->w, prev->numOut, 
                    0, dy, prev->numOut);

        if(binIn){
            for(int i = 0; i < size * numIn; i++){
                dy[i] = get_sigmoid_derivative(y[i]) * dy[i];
            }
        }
    }
}

void EncoderLayer::getHDeriv(EncoderLayer* prev, int size){
    if(prev == this){
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, numOut, numIn,
                    1, dy, numIn, w, numOut, 
                    0, dh, numOut);

        if(binOut){
            for(int i = 0; i < size * numOut; i++){
                dh[i] = get_sigmoid_derivative(h[i]) * dh[i];
            }
        }
    }else{
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    size, prev->numIn, prev->numOut,
                    1, prev->dh, prev->numOut, prev->w, prev->numOut, 
                    0, dh, prev->numIn);

        if(binOut){
            for(int i = 0; i < size * numOut; i++){
                dh[i] = get_sigmoid_derivative(h[i]) * dh[i];
            }
        }
    }
}

void EncoderLayer::getDeltaFromYDeriv(double* prevY, int size){
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numIn, numOut, size,
                -1.0 * lr / size, dy, numIn, prevY, numOut, 
                0, dw, numOut);

    // update y bias
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numIn, 1, size,
                -1.0 * lr / size, dy, numIn, 
                I(), 1, 1, c, 1);
}

void EncoderLayer::getDeltaFromHDeriv(int size){
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numIn, numOut, size,
                -1.0 * lr / size, x, numIn, dh, numOut, 
                1, dw, numOut);

    // update h bias
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numOut, 1, size,
                -1.0 * lr / size, dh, numOut, 
                I(), 1, 1, b, 1);

    // update w
    cblas_daxpy(numIn*numOut, 1, dw, 1, w, 1);
}


DeepAutoEncoder::DeepAutoEncoder() : UnsuperviseTrainComponent("DeepAutoEncoder")
{
    
}

DeepAutoEncoder::DeepAutoEncoder(int n, int* sizes) : 
    UnsuperviseTrainComponent("DeepAutoEncoder"), numLayer(n)
{
    for(int i = 0; i < n; i++){
        layers[i] = new EncoderLayer(sizes[i], sizes[i+1]);
    }
    layers[n-1]->binOut = false;
}

DeepAutoEncoder::DeepAutoEncoder(MultiLayerRBM& multirbm):
    UnsuperviseTrainComponent("DeepAutoEncoder")
{
    numLayer = multirbm.getLayerNumber();
    for(int i = 0; i < numLayer; i++){
        int nIn = multirbm[i]->getInputNumber();
        int nOut = multirbm[i]->getOutputNumber();
        double* trans = new double[nIn*nOut];
        multirbm[i]->getWeightTrans(trans);
        layers[i] = new EncoderLayer(nIn, nOut, trans, 
                multirbm[i]->getHBias(), multirbm[i]->getVBias());
        delete[] trans;
    }
}

DeepAutoEncoder::~DeepAutoEncoder(){
    for(int i = 0; i < numLayer; i++)
        delete layers[i];
}

void DeepAutoEncoder::beforeTraining(int size){
    for(int i = 0; i < numLayer; i++){
        layers[i]->allocate(size);
    }
}

void DeepAutoEncoder::trainBatch(int size){
    forward(size);
    backpropagate(size);
}

void DeepAutoEncoder::runBatch(int size){
    forward(size);
}

void DeepAutoEncoder::setLearningRate(double lr){
    for(int i = 0; i < numLayer; i++){
        layers[i]->lr = lr;
    }
}

void DeepAutoEncoder::setInput(double *input){
    layers[0]->setInput(input);
}

void DeepAutoEncoder::setLabel(double *label){}

int DeepAutoEncoder::getInputNumber(){
    return layers[0]->numIn;
}

double* DeepAutoEncoder::getOutput(){
    return layers[numLayer-1]->h;
}

int DeepAutoEncoder::getOutputNumber(){
    return layers[numLayer-1]->numOut; 
}

double* DeepAutoEncoder::getLabel() { return NULL; }

double DeepAutoEncoder::getReconstructCost(double *x, double *y, int numIn, int size){
    double res;

    for(int i = 0; i < numIn * size; i++){
        temp[i] = x[i] >= 1.0 ? 1.0 : x[i];
        temp2[i] = log(y[i] + 1e-10);
    }
    res = cblas_ddot(size * numIn, temp, 1, temp2, 1);

    for(int i = 0; i < numIn * size; i++){
        temp[i] = 1.0 - x[i] >= 1.0 ? 1.0 : 1.0 - x[i];
        temp2[i] = log(1.0 - y[i] + 1e-10);
    }

    res += cblas_ddot(size * numIn, temp, 1, temp2, 1);
    return -1.0 * res / size;
}

double DeepAutoEncoder::getTrainingCost(int size, int numBatch){
    return getReconstructCost(layers[0]->x, layers[0]->y, layers[0]->numIn, size);
}

void DeepAutoEncoder::saveModel(FILE*){};
void DeepAutoEncoder::operationPerEpoch(){}

void DeepAutoEncoder::forward(int size){
    for(int i = 0; i < numLayer; i++){
        if(i != 0){
            layers[i]->setInput(layers[i-1]->h);
        }
        layers[i]->getHFromX(layers[i]->x, layers[i]->h, size);
    }
    for(int i = numLayer-1; i >= 0; i--){
        if(i == numLayer-1){
            layers[i]->getYFromH(layers[i]->h, layers[i]->y, size);
        }else{
            layers[i]->getYFromH(layers[i+1]->y, layers[i]->y, size);
        }
    }
}

void DeepAutoEncoder::backpropagate(int size){
    for(int i = 0; i < numLayer; i++){
        if(i == 0){
            layers[i]->getYDeriv(NULL, size);
        }else{
            layers[i]->getYDeriv(layers[i-1], size);
        }

        if(i != numLayer-1){
            layers[i]->getDeltaFromYDeriv(layers[i+1]->y, size);
        }else{
            layers[i]->getDeltaFromYDeriv(layers[i]->h, size);
        }
    }

    for(int i = numLayer-1; i >= 0; i--){
        if(i == numLayer-1){
            layers[i]->getHDeriv(layers[i], size);
        }else{
            layers[i]->getHDeriv(layers[i+1], size);
        }

        layers[i]->getDeltaFromHDeriv(size);
    }
}


