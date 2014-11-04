extern "C" {
#include "cblas.h"
}
#include "Config.h"
#include "Dataset.h"
#include "Utility.h"
#include <cstring>
#include <ctime>

#ifndef _RBM_H
#define _RBM_H

class RBM {
    public:
        RBM(int, int);
        ~RBM();
        void setBuffer(double *v2, double *h1, double *h2, 
                       double *pv, double *ph1, double *ph2);
        void train(Dataset *data, double learningRate = 0.1, int batchSize = 20, int numEpoch = 15);

        void setWeightFile(const char *);
        void dumpWeight(int, int);

    private:
        void getHProb(const double *v, double *ph, const int size);
        void getHSample(const double *ph, double *h, const int size);
        void getVProb(const double *h, double *pv, const int size);
        void getVSample(const double *pv, double *v, const int size);
        void gibbsSampleHVH(const double *hStart, double *h, double *ph, 
                                  double *v, double *pv, const int step, const int size);
        void getFE(const double *v, double *FE, const int size);
        double getPL(double *v, const int size);

        int numVis, numHid;
        double *weight;
        double *hbias, *vbias;
        double *v1, *v2, *h1, *h2;
        double *ph1, *ph2, *pv;

        char *weightFile;
};

#endif
