extern "C" {
#include "cblas.h"
}
#include "Config.h"
#include "Dataset.h"
#include "Utility.h"
#include "TrainModel.h"
#include <cstring>
#include <ctime>
#include <cstdio>

#ifndef _RBM_H
#define _RBM_H

class MultiLayerRBM;

class RBM : public UnsuperviseTrainComponent {
    public:
        RBM(int, int);
        RBM(const char *);
        RBM(FILE *);
        ~RBM();

        void setWeightFile(const char *);
        void dumpWeight(int, int);

        void beforeTraining(int);
        void afterTraining(int);
        void trainBatch(int);
        void runBatch(int);
        void setLearningRate(double lr);
        void setInput(double *input);
        double* getOutput() { return ph1; }
        int getOutputNumber() { return numHid; }
        double getTrainingCost(int, int);
        void operationPerEpoch();

        void saveModel(FILE* modelFileFd);
    private:
        void getHProb(const double *v, double *ph, const int size);
        void getHSample(const double *ph, double *h, const int size);
        void getVProb(const double *h, double *pv, const int size);
        void getVSample(const double *pv, double *v, const int size);
        void gibbsSampleHVH(const double *hStart, double *h, double *ph, 
                                  double *v, double *pv, const int step, const int size);
        void getFE(const double *v, double *FE, const int size);
        double getPL(double *v, const int size);

        void runChain(int, int);
        void updateWeight(int);
        void updateBias(int);

        void loadModel(FILE*);
        void allocateBuffer(int);
        void freeBuffer();

        int numVis, numHid;
        double *weight;
        double *hbias, *vbias;
        double *v1, *v2, *h1, *h2;
        double *ph1, *ph2, *pv;
        double *chainStart;

        double learningRate;

        char *weightFile;
        friend class MultiLayerRBM;
};

#endif
