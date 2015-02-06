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
        int getInputNumber();
        int getOutputNumber();
        double* getLabel();
        void saveModel(FILE*);
        int getLayerNumber() { return numLayer; }
        MLPLayer* getLayer(int i) { return layers[i]; }

        void setLayerNumber(int n) { numLayer = n; }

        inline void addLayer(MLPLayer* l) { layers[numLayer++] = l; }
        inline void setLayer(int i, MLPLayer* l) { layers[i] = l; }

        void setGaussian(bool b) { gaussian = b; }
        void operationPerEpoch(int k);
    private:
        MLPLayer* layers[maxLayer];
        int numLayer;
        void loadModel(FILE*);

        double learningRate;
        double *label;

        bool gaussian;
};

#endif
