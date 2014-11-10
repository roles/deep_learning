#include "Dataset.h"
#include "TrainModel.h"
#include "MLPLayer.h"
#include "Logistic.h"

#ifndef _MLP_H
#define _MLP_H

class MLP : public TrainComponent{
    public:
        MLP();
        MLP(const char*);
        ~MLP();
        void trainBatch(int);
        void runBatch(int);
        void setLearningRate(double lr);
        void setInput(double *input);
        void setLabel(double *label);
        double* getOutput();
        int getOutputNumber();
        double* getLabel();
        void saveModel(FILE*);

        inline void addLayer(MLPLayer* l) { layers[numLayer++] = l; }
    private:
        MLPLayer* layers[maxLayer];
        int numLayer;
        void loadModel(FILE*);

        double learningRate;
        double *label;
};

#endif
