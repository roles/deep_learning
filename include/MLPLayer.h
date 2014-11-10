extern "C"{
#include "cblas.h"
}
#include "Config.h"
#include "Utility.h"
#include <cstring>
#include <cstdio>

#ifndef _MLPLAYER_H
#define _MLPLAYER_H

enum UnitType { Sigmoid, Tanh , Softmax };

typedef double MLPBuffer[maxUnit*maxUnit];

class MLPLayer {
    public:
        MLPLayer(int, int, const char* name = "MLPLayer"); 
        MLPLayer(FILE* modelFileFd, const char *name = "MLPLayer");
        MLPLayer(const char*, const char *name = "MLPLayer");
        virtual ~MLPLayer();
        virtual void forward(int);     
        virtual void backpropagate(int size, MLPLayer *prevLayer);

        inline void setLearningRate(double lr) { learningRate = lr; }
        inline void setInput(double* input) { in = input; }
        inline int getInputNumber() { return numIn; }
        inline int getOutputNumber() { return numOut; }
        inline double* getDelta() { return delta; }
        inline double* getWeight() { return weight; }
        inline double* getBias() { return bias; }
        double* getOutput();
        inline void setUnitType(UnitType t) { unitType = t; }

        virtual void setLabel(double* label) { }    //用于最后一层有监督训练
        virtual double* getLabel() { return NULL; }

        virtual void saveModel(FILE* modelFileFd);

        const char* getLayerName() { return layerName; }
    protected:
        void initializeWeight(){ }  //构造函数中不能调用虚函数
        void initializeBias(){ memset(bias, 0, numOut*sizeof(double)); }
    private:
        virtual void computeDelta(int, MLPLayer*);
        virtual void updateWeight(int);
        virtual void updateBias(int);
        virtual void loadModel(FILE* modelFileFd);

        virtual void computeNeuron(int) = 0;
        virtual void computeNeuronDerivative(double*, int) = 0;
        void init();
        int numIn, numOut;
        double *weight, *delta, *bias;
        UnitType unitType;

        double *in, *out;
        double learningRate;
        char layerName[20];
};

class SigmoidLayer : public MLPLayer {
    public:
        SigmoidLayer(int, int); 
        SigmoidLayer(FILE* modelFileFd);
        SigmoidLayer(const char*);
    private:
        void computeNeuron(int);
        void computeNeuronDerivative(double*, int);
        void initializeWeight();
};

class TanhLayer : public MLPLayer {
    public:
        TanhLayer(int, int); 
        TanhLayer(FILE* modelFileFd);
        TanhLayer(const char*);
    private:
        void computeNeuron(int);
        void computeNeuronDerivative(double*, int);
        void initializeWeight();
};

class SoftmaxLayer : public MLPLayer {
    public:
        SoftmaxLayer(int, int, const char* name = "Softmax"); 
        SoftmaxLayer(FILE* modelFileFd, const char* name = "Softmax");
        SoftmaxLayer(const char*, const char* name = "Softmax");
    private:
        void computeNeuron(int);
        void computeNeuronDerivative(double*, int);
        void initializeWeight();
};

#endif
