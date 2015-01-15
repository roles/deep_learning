#include "TrainComponent.h"
#include <cstring>
#include "Dataset.h"
#include "TrainModel.h"
#include "Config.h"
#include "mkl_cblas.h"
#include "Utility.h"

static double temp[maxUnit*maxUnit];
static double temp2[maxUnit*maxUnit];

class DeepAutoEncoder;
class EncoderLayer;

class EncoderLayer {
    public:
        EncoderLayer(int, int);
        ~EncoderLayer();
        void setInput(double *input) { x = input; }
        void allocate(int);
    private:
        int numIn, numOut;
        double *x, *y, *h;
        double *w, *b, *c;
        double *dw;
        double *dy, *dh;
        double lr;

        void getHFromX(double*, double*, int);
        void getYFromH(double*, double*, int);
        void getYDeriv(EncoderLayer* prev, int);
        void getHDeriv(EncoderLayer* prev, int);
        void getDeltaFromYDeriv(double*, int size);
        void getDeltaFromHDeriv(int size);

        friend class DeepAutoEncoder;
};

EncoderLayer::EncoderLayer(int numIn, int numOut) :
    numIn(numIn), numOut(numOut), y(NULL), h(NULL),
    dy(NULL), dh(NULL)
{
    w = new double[numIn*numOut];
    dw = new double[numIn*numOut];
    b = new double[numOut];
    c = new double[numIn];
    memset(b, 0, numOut*sizeof(double));
    memset(c, 0, numIn*sizeof(double));
    initializeWeightSigmoid(w, numIn, numOut);
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

    for(int i = 0; i < size * numOut; i++){
        h[i] = sigmoid(h[i]);
    }
}

void EncoderLayer::getYFromH(double *h, double *y, int size){
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

void EncoderLayer::getYDeriv(EncoderLayer* prev, int size){
    if(prev == NULL){
        for(int i = 0; i < size * numIn; i++){
            dy[i] = y[i] - x[i];
        }
    }else{
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, prev->numOut, prev->numIn,
                    1, prev->dy, prev->numIn, prev->w, prev->numOut, 
                    0, dy, prev->numOut);

        for(int i = 0; i < size * numIn; i++){
            dy[i] = get_sigmoid_derivative(y[i]) * dy[i];
        }
    }
}

void EncoderLayer::getHDeriv(EncoderLayer* prev, int size){
    if(prev == this){
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, numOut, numIn,
                    1, dy, numIn, w, numOut, 
                    0, dh, numOut);

        for(int i = 0; i < size * numOut; i++){
            dh[i] = get_sigmoid_derivative(h[i]) * dh[i];
        }
    }else{
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    size, prev->numIn, prev->numOut,
                    1, prev->dh, prev->numOut, prev->w, prev->numOut, 
                    0, dh, prev->numIn);

        for(int i = 0; i < size * numOut; i++){
            dh[i] = get_sigmoid_derivative(h[i]) * dh[i];
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

class DeepAutoEncoder : public UnsuperviseTrainComponent{
    public:
        DeepAutoEncoder();
        DeepAutoEncoder(int, int*);
        ~DeepAutoEncoder();
        void beforeTraining(int size);
        void trainBatch(int);
        void runBatch(int);
        void setLearningRate(double lr);
        void setInput(double *input);
        void setLabel(double *label);
        int getInputNumber();
        double* getOutput();
        int getOutputNumber();
        double* getLabel();
        double getTrainingCost(int, int);
        void saveModel(FILE*);
        void operationPerEpoch();

        void addLayer(int, int);
        void forward(int);
        void backpropagate(int);
    private:
        double getReconstructCost(double *x, double *y, int n, int size);
        int numLayer;
        EncoderLayer* layers[maxLayer];
};

DeepAutoEncoder::DeepAutoEncoder() : UnsuperviseTrainComponent("DeepAutoEncoder")
{
    
}

DeepAutoEncoder::DeepAutoEncoder(int n, int* sizes) : 
    UnsuperviseTrainComponent("DeepAutoEncoder"), numLayer(n)
{
    for(int i = 0; i < n; i++){
        layers[i] = new EncoderLayer(sizes[i], sizes[i+1]);
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

int main(){
    MNISTDataset data;
    data.loadData();

    int sizes[] = {data.getFeatureNumber(), 500, 500};

    DeepAutoEncoder dad(2, sizes);
    TrainModel model(dad);
    model.train(&data, 0.1, 20, 15);
}

