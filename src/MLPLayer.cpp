#include "MLPLayer.h"
#include "mkl_cblas.h"

static MLPBuffer temp, temp2;

MLPLayer::MLPLayer(int nIn, int nOut, const char* name) :
    numIn(nIn), numOut(nOut), out(NULL), delta(NULL)
{
    strcpy(layerName, name);
    init(); 
}

MLPLayer::MLPLayer(FILE* fd, const char* name) :
    out(NULL), delta(NULL)
{
    strcpy(layerName, name);
    loadModel(fd);
}

MLPLayer::MLPLayer(const char* file, const char* name) :
    out(NULL), delta(NULL)
{
    FILE* fd = fopen(file, "rb");
    strcpy(layerName, name);
    loadModel(fd);
    fclose(fd);
}

MLPLayer::MLPLayer(int nIn, int nOut, double* weight, double* bias, const char *name) :
    numIn(nIn), numOut(nOut), out(NULL), delta(NULL)
{
    strcpy(layerName, name);
    init();
    memcpy(this->weight, weight, sizeof(double)*numIn*numOut);
    memcpy(this->bias, bias, sizeof(double)*numOut);
}


void MLPLayer::loadModel(FILE* fd){
    fread(&numIn, sizeof(int), 1, fd);
    fread(&numOut, sizeof(int), 1, fd);
    printf("numIn : %d\nnumOut : %d\n", numIn, numOut);

    weight = new double[numIn*numOut];
    bias = new double[numOut];

    for(int i = 0; i < numIn*numOut; i++){
        fread(&weight[i], sizeof(double), 1, fd);
    }
    for(int i = 0; i < numOut; i++){
        fread(&bias[i], sizeof(double), 1, fd);
    }
}

double* MLPLayer::getOutput() { return out; }

void MLPLayer::init(){
    weight = new double[numIn*numOut];
    bias = new double[numOut];
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

    computeNeuron(size);
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
    computeNeuronDerivative(deriv, size);

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

void MLPLayer::saveModel(FILE *fd){
    fwrite(&numIn, sizeof(int), 1, fd);
    fwrite(&numOut, sizeof(int), 1, fd);
    for(int i = 0; i < numIn*numOut; i++){
        fwrite(&weight[i], sizeof(double), 1, fd);
    }
    for(int i = 0; i < numOut; i++){
        fwrite(&bias[i], sizeof(double), 1, fd);
    }
}

SigmoidLayer::SigmoidLayer(int numIn, int numOut) :
    MLPLayer(numIn, numOut, "Sigmoid")
{
    initializeWeight();
    MLPLayer::initializeBias();
}
SigmoidLayer::SigmoidLayer(FILE* modelFileFd) :
    MLPLayer(modelFileFd, "Sigmoid") { }
SigmoidLayer::SigmoidLayer(const char* file) :
    MLPLayer(file, "Sigmoid") { }
SigmoidLayer::SigmoidLayer(int nIn, int nOut, double* weight, double* bias) :
    MLPLayer(nIn, nOut, weight, bias, "Sigmoid") { }

void SigmoidLayer::computeNeuron(int size){
    double *out = getOutput();
    int numOut = getOutputNumber();
    for(int i = 0; i < size*numOut; i++){
        //out[i] = sigmoid(out[i]);
        out[i] = sigmoidc(out[i]);
    }
}

void SigmoidLayer::computeNeuronDerivative(double* deriv, int size){
    double *out = getOutput();
    int numOut = getOutputNumber();
    for(int i = 0; i < size*numOut; i++){
        deriv[i] = get_sigmoid_derivative(out[i]);
    }
}

void SigmoidLayer::initializeWeight(){
    initializeWeightSigmoid(getWeight(), getInputNumber(), getOutputNumber());
}

TanhLayer::TanhLayer(int numIn, int numOut) :
    MLPLayer(numIn, numOut, "Tanh")
{ 
    initializeWeight();
    MLPLayer::initializeBias();
}
TanhLayer::TanhLayer(FILE* modelFileFd) :
    MLPLayer(modelFileFd, "Tanh") { }
TanhLayer::TanhLayer(const char* file) :
    MLPLayer(file, "Tanh") { }
TanhLayer::TanhLayer(int nIn, int nOut, double* weight, double* bias) :
    MLPLayer(nIn, nOut, weight, bias, "Tanh") { }

void TanhLayer::computeNeuron(int size){
    double *out = getOutput();
    int numOut = getOutputNumber();
    for(int i = 0; i < size*numOut; i++){
        out[i] = tanh(out[i]);
    }
}

void TanhLayer::computeNeuronDerivative(double* deriv, int size){
    double *out = getOutput();
    int numOut = getOutputNumber();
    for(int i = 0; i < size*numOut; i++){
        deriv[i] = get_tanh_derivative(out[i]);
    }
}

void TanhLayer::initializeWeight(){
    initializeWeightTanh(getWeight(), getInputNumber(), getOutputNumber());
}

SoftmaxLayer::SoftmaxLayer(int numIn, int numOut, const char* name) :
    MLPLayer(numIn, numOut, name)
{
    initializeWeight();
    MLPLayer::initializeBias();
}
SoftmaxLayer::SoftmaxLayer(FILE* modelFileFd, const char* name) :
    MLPLayer(modelFileFd, name) { }
SoftmaxLayer::SoftmaxLayer(const char* file, const char* name) :
    MLPLayer(file, name) { }
SoftmaxLayer::SoftmaxLayer(int nIn, int nOut, double* weight, double* bias, const char* name) :
    MLPLayer(nIn, nOut, weight, bias, name) { }

void SoftmaxLayer::computeNeuron(int size){
    double *out = getOutput();
    int numOut = getOutputNumber();
    for(int i = 0; i < size; i++){
        softmax(out+i*numOut, numOut);
    }
}

void SoftmaxLayer::computeNeuronDerivative(double* deriv, int size){ }

void SoftmaxLayer::initializeWeight(){
    memset(getWeight(), 0, getInputNumber()*getOutputNumber()*sizeof(double));
}
