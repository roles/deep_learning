#include "TrainComponent.h"

#ifndef _CLASSRBM_H
#define _CLASSRBM_H

/*
 * E(x,y,h) = h'Wx + b'x + c'h + d'y + h'Uy
 */
class ClassRBM : public TrainComponent {
    public:
        void beforeTraining(int);
        void afterTraining(int);
        void trainBatch(int);
        void runBatch(int);
        void setLearningRate(double lr) { learningRate = lr; }
        void setInput(double *input) { x = input; }
        void setLabel(double *label) { y = label; }
        double* getOutput() { return py; }
        int getOutputNumber() { return numLabel; }
        double* getLabel() { return y; }

        void saveModel(FILE* fd);

        ClassRBM(int, int, int);
        ~ClassRBM();
    private:
        int numVis, numHid, numLabel;
        double *W, *U, *b, *c, *d;
        double *x, *y, *h;
        double *ph, *py;
        double *phk;    // all possible ph

        double learningRate;

        void getHProb(double* x, double* y, double* ph, int size);
        void getYProb(double* x, double* py, int size);
        void update(int);
        void updateW(int);
        void updateU(int);
        void updateYBias(int);
        void updateHbias(int);
        void initBuffer(int);
        void forward(int);
};

#endif
