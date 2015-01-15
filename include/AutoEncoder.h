#include "TrainComponent.h"

#ifndef _AUTOENCODER_H
#define _AUTOENCODER_H

class AutoEncoder : public UnsuperviseTrainComponent{
    public:
        AutoEncoder(int, int, bool = false);
        ~AutoEncoder();
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
        void saveModel(FILE*){};
        void operationPerEpoch();
    private:
        int numIn, numOut;
        double *x, *y, *h;
        double *nx;
        double *w, *b, *c;
        double lr;
        bool denoise;

        void getHFromX(double*, double*, int);
        void getYFromH(double*, double*, int);
        void backpropagate(int size);
        double getReconstructCost(double *x, double *y, int size);
        void dumpWeight(int, int, const char* = "result/da_weight.txt");
};

#endif
