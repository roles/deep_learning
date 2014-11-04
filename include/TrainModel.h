#include "Dataset.h"

#ifndef _TRAINMODEL_H
#define _TRAINMODEL_H

enum TrainType { Unsupervise, Supervise };

class TrainComponent {
    public:
        TrainComponent(TrainType t);
        virtual void trainBatch(int) = 0;   // forward + backpropagate
        virtual void runBatch(int) = 0;     // forward only
        virtual void setLearningRate(double lr) = 0;
        virtual void setInput(double *input) = 0;
        virtual void setLabel(double *label) = 0;
        virtual double* getOutput() = 0;
        virtual double* getLabel() = 0;
        inline TrainType getTrainType() { return trainType; }
        virtual ~TrainComponent();
    private:
        TrainType trainType;
};

class TrainModel {
    public:
        TrainModel(TrainComponent& comp);
        void train(Dataset *, double, int, int);
        double getValidError(Dataset *, int);
    private:
        TrainComponent& component;
};

#endif
