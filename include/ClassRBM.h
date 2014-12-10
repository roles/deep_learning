#include "TrainComponent.h"

#ifndef _CLASSRBM_H
#define _CLASSRBM_H

/*
 * E(x,y,h) = h'Wx + b'x + c'h + d'y + h'Uy
 */
class ClassRBM : public TrainComponent {
    public:
        using IModel::saveModel;    // 名称遮掩，查看effective c++相关章节
        void beforeTraining(int);
        void trainBatch(int);
        void runBatch(int);
        void setLearningRate(double lr) { learningRate = lr; }
        void setInput(double *input) { x = input; }
        void setLabel(double *label) { y = label; }
        double* getOutput() { return py; }
        int getOutputNumber() { return numLabel; }
        double* getTransOutput() { return ph; }
        int getTransOutputNumber() { return numHid; }
        double* getLabel() { return y; }

        void saveModel(FILE* fd);

        void resampleFromH(const char resampleFile[] = "result/resample.bin");

        ClassRBM(int, int, int);
        ClassRBM(const char*);
        ~ClassRBM();
    private:
        int numVis, numHid, numLabel;
        double *W, *U, *b, *c, *d;
        double *x, *y, *h;
        double *ph, *py;
        double *phk;    // all possible ph

        double *yGen, *xGen;

        double learningRate;
        double alpha;   // hyrid rate

        void getHProb(double* x, double* y, double* ph, int size);
        void getYProb(double* x, double* py, int size);
        void getYFromH(double* h, double *y, int size);
        void getXFromH(double* h, double *x, int size);
        void getYProbFromH(double* h, double *py, int size);
        void getXProbFromH(double* h, double *px, int size);
        void update(int);
        void updateW(int);
        void updateU(int);
        void updateYBias(int);
        void updateHbias(int);
        void updateXBias(int);
        void initBuffer(int);
        void forward(int);
        void loadModel(FILE*);
        void resampleUnitFromH(FILE* fd, double* h);
};

#endif
