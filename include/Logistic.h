#include "Config.h"
#include "TrainModel.h"
#include "Utility.h"
#include "Dataset.h"
#include "MLPLayer.h"
#include <ctime>
#include <cstring>

#ifndef _LOGISTIC_H
#define _LOGISTIC_H

class Logistic : public SoftmaxLayer , public TrainComponent 
{
    public:
        Logistic(int, int);
        Logistic(FILE *fd);
        Logistic(const char*);
        void trainBatch(int);   // forward + backpropagate
        void runBatch(int);     // forward only
        void setLearningRate(double lr);
        void setInput(double *input);
        void setLabel(double *label);
        double* getOutput();
        int getOutputNumber();
        double* getLabel();
        void saveModel(FILE* modelFileFd);
    private:
        void computeDelta(int, MLPLayer*);
        double getValidError(Dataset*, int);

        double *label;
};

#endif
