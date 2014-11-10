#include <cstdio>

#ifndef _TRAINCOMPONENT_H
#define _TRAINCOMPONENT_H

enum TrainType { Unsupervise, Supervise };

class IModel {
    public:
        IModel(const char* name);
        virtual ~IModel();
        virtual void saveModel(FILE* fd) = 0;
        void setModelFile(const char *);
        virtual void saveModel();
        inline const char* getModelFile() { return modelFile; }
        const char* getModelName() { return modelName; }
    private:
        char *modelFile;
        char modelName[20];
};

class TrainComponent : public IModel {
    public:
        TrainComponent(TrainType t, const char* name);
        virtual void beforeTraining(int);
        virtual void afterTraining(int);
        virtual void trainBatch(int) = 0;   // forward + backpropagate
        virtual void runBatch(int) = 0;     // forward only
        virtual void setLearningRate(double lr) = 0;
        virtual void setInput(double *input) = 0;
        virtual void setLabel(double *label) = 0;
        virtual double* getOutput() = 0;
        virtual int getOutputNumber() = 0;
        virtual double* getLabel() = 0;
        virtual double getTrainingCost(int, int) { return 0.0; }
        virtual void operationPerEpoch() { }
        inline TrainType getTrainType() { return trainType; }
        virtual ~TrainComponent();

    private:
        TrainType trainType;
};

class UnsuperviseTrainComponent : public TrainComponent {
    public:
        UnsuperviseTrainComponent(const char* name);
        void setLabel(double *label) { }
        double* getLabel() { return NULL; }
        virtual ~UnsuperviseTrainComponent();
};

#endif
