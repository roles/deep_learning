#include "TrainComponent.h"
#include "Config.h"
#include "MultiLayerRBM.h"

#ifndef _DEEPAUTOENCODER_H
#define _DEEPAUTOENCODER_H

class DeepAutoEncoder;
class EncoderLayer;

class EncoderLayer {
    public:
        EncoderLayer(int, int);
        EncoderLayer(int numIn, int numOut, double *w, double *b, double *c);
        ~EncoderLayer();
        void setInput(double *input) { x = input; }
        void allocate(int);
        int getInputNumber() { return numIn; }
        int getOutputNumber() { return numIn; }
        double getInputBias() { return c; }
        double getOutputBias() { return b; }
    private:
        int numIn, numOut;
        double *x, *y, *h;
        double *w, *b, *c;
        double *dw;
        double *dy, *dh;
        double lr;

        bool binIn, binOut;     // whether is binary input, binary output

        void getHFromX(double*, double*, int);
        void getYFromH(double*, double*, int);
        void getYDeriv(EncoderLayer* prev, int);
        void getHDeriv(EncoderLayer* prev, int);
        void getDeltaFromYDeriv(double*, int size);
        void getDeltaFromHDeriv(int size);

        void getWeightTrans(double* transWeight);

        friend class DeepAutoEncoder;
};

class DeepAutoEncoder : public UnsuperviseTrainComponent{
    public:
        DeepAutoEncoder();
        DeepAutoEncoder(int, int*);
        DeepAutoEncoder(MultiLayerRBM&);
        DeepAutoEncoder(const char*);
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

        int getLayerNumber() { return numLayer; }
        EncoderLayer* getLayer(int i) { return layers[i]; }
    private:
        double getReconstructCost(double *x, double *y, int n, int size);
        int numLayer;
        EncoderLayer* layers[maxLayer];
};
#endif
