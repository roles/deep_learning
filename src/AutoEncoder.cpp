#include "Config.h"
#include "Utility.h"
#include "mkl_cblas.h"
#include <cstring>
#include "AutoEncoder.h"

static double temp[maxUnit*maxUnit], temp2[maxUnit*maxUnit];
static double temp3[maxUnit*maxUnit];

void AutoEncoder::beforeTraining(int size){
    h = new double[size*numOut];
    y = new double[size*numIn];
    if(denoise){
        nx = new double[size*numIn];
    }
}

AutoEncoder::AutoEncoder(int numIn, int numOut, bool denoise) :
    numIn(numIn), numOut(numOut), denoise(denoise),
    UnsuperviseTrainComponent("AutoEncoder")
{
    w = new double[numIn*numOut];    
    b = new double[numOut];
    c = new double[numIn];
    initializeWeightSigmoid(w, numIn, numOut);
    memset(b, 0, sizeof(double)*numOut);
    memset(c, 0, sizeof(double)*numIn);
}

AutoEncoder::~AutoEncoder(){
    delete[] w;
    delete[] b;
    delete[] c;
    delete[] y;
    delete[] h;
    if(denoise) delete[] nx;
}

void AutoEncoder::trainBatch(int size){
    if(denoise){
        corrupt(x, nx, size*numIn, 0.3);
    }else{
        nx = x;
    }
    getHFromX(nx, h, size);
    getYFromH(h, y, size);
    backpropagate(size);
}

void AutoEncoder::runBatch(int size){
    getHFromX(x, h, size);
    getYFromH(h, y, size);
}

void AutoEncoder::setLearningRate(double lr){
    this->lr = lr;
}

void AutoEncoder::setInput(double *input){
    x = input;
}

void AutoEncoder::setLabel(double *label){ }

int AutoEncoder::getInputNumber(){
    return numIn;
}

double* AutoEncoder::getOutput(){
    return h;
}

int AutoEncoder::getOutputNumber(){
    return numOut;
}

double* AutoEncoder::getLabel(){
    return NULL;
}

double AutoEncoder::getReconstructCost(double *x, double *y, int size){
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

double AutoEncoder::getTrainingCost(int size, int numBatch){
    return getReconstructCost(x, y, size);
}

void AutoEncoder::getHFromX(double *x, double *h, int size){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, numOut, numIn,
                1, x, numIn, w, numOut,
                0, h, numOut);

    cblas_dger(CblasRowMajor, size, numOut,
               1.0, I(), 1, b, 1, h, numOut);

    for(int i = 0; i < size * numOut; i++){
        h[i] = sigmoid(h[i]);
    }
}

void AutoEncoder::getYFromH(double *h, double *y, int size){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, numIn, numOut,
                1, h, numOut, w, numOut,
                0, y, numIn);

    cblas_dger(CblasRowMajor, size, numIn,
               1.0, I(), 1, c, 1, y, numIn);

    for(int i = 0; i < size * numIn; i++){
        y[i] = sigmoid(y[i]);
    }
}

void AutoEncoder::backpropagate(int size){
    double* xydiff = temp;
    for(int i = 0; i < size * numIn; i++){
        xydiff[i] = y[i] - x[i];
    }

    // update dc = y-x
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numIn, 1, size,
                -1.0 * lr / size, xydiff, numIn, 
                I(), 1, 1, c, 1);

    double* delta = temp3;
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numIn, numOut, size,
                1, xydiff, numIn, h, numOut, 
                0, delta, numOut);

    double* hdelta = temp2;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, numOut, numIn,
                1, xydiff, numIn, w, numOut,
                0, hdelta, numOut);

    for(int i = 0; i < size * numOut; i++){
        hdelta[i] = get_sigmoid_derivative(h[i]) * hdelta[i];
    }

    // update db = sum(y-x)*w*(1-h)*h
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numOut, 1, size,
                -1.0 * lr / size, hdelta, numOut, 
                I(), 1, 1, b, 1);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numIn, numOut, size,
                1, nx, numIn, hdelta, numOut, 
                1, delta, numOut);

    // update weight
    cblas_daxpy(numIn*numOut, -1.0 * lr / size, delta, 1, w, 1);
}

void AutoEncoder::operationPerEpoch(){
    dumpWeight(100, 28);
}

void AutoEncoder::dumpWeight(int numDumpHid, int numVisPerLine, const char* weightFile){
    if(weightFile == NULL) return;
    FILE *weightFd = fopen(weightFile, "a+");

    for(int i = 0; i < numDumpHid; i++){
        for(int j = 0; j < numIn; j++){
            fprintf(weightFd, "%.5lf%s", w[j*numOut+i], 
                ((j+1) % numVisPerLine == 0) ? "\n" : "\t");
        }
    }

    fclose(weightFd);
}

