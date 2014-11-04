extern "C"{
#include "cblas.h"
}
#include "Config.h"
#include "Utility.h"
#include <cstring>

#ifndef _MLPLAYER_H
#define _MLPLAYER_H

enum UnitType { Sigmoid, Tanh , Softmax };

typedef double MLPBuffer[maxUnit*maxBatchSize];

static MLPBuffer temp, temp2;

class MLPLayer {
    public:
        MLPLayer(int, int); 
        MLPLayer(int, int, UnitType); 
        ~MLPLayer();
        virtual void forward(int);     
        virtual void backpropagate(int size, MLPLayer *prevLayer);

        inline void setLearningRate(double lr) { learningRate = lr; }
        inline void setInput(double* input) { in = input; }
        inline int getOutputNumber() {return numOut;}
        inline double* getDelta() { return delta; }
        inline double* getWeight() { return weight; }
        double* getOutput();
        inline void setUnitType(UnitType t) { unitType = t; }

        virtual void setLabel(double* label) { }    //用于最后一层有监督
        virtual double* getLabel() { return NULL; }
    private:
        virtual void computeDelta(int, MLPLayer*);
        virtual void updateWeight(int);
        virtual void updateBias(int);
        void init();
        int numIn, numOut;
        double *weight, *delta, *bias;
        UnitType unitType;

        double *in, *out;
        double learningRate;
};

#endif
