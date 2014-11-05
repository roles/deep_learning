#include "RBM.h"

static double temp[maxUnit*maxUnit];

RBM::RBM(int numVis, int numHid)
    : numVis(numVis), numHid(numHid),
      weightFile(NULL)
{
    weight = new double[numVis*numHid];
    initializeWeightSigmoid(weight, numVis, numHid);

    vbias = new double[numVis];
    hbias = new double[numHid];
    memset(vbias, 0, numVis*sizeof(double));
    memset(hbias, 0, numHid*sizeof(double));
}

RBM::~RBM(){
    delete[] weight;
    delete[] vbias;
    delete[] hbias;
}

void RBM::setBuffer(double *v2, double *h1, double *h2, 
                    double *pv, double *ph1, double *ph2){
    this->v2 = v2;
    this->h1 = h1;
    this->h2 = h2;
    this->pv = pv;
    this->ph1 = ph1;
    this->ph2 = ph2;
}

void RBM::setWeightFile(const char *weightFile){
    this->weightFile = new char[strlen(weightFile)+1];
    strcpy(this->weightFile, weightFile);
}

void RBM::dumpWeight(int numDumpHid, int numVisPerLine){
    if(weightFile == NULL) return;
    FILE *weightFd = fopen(weightFile, "a+");

    for(int i = 0; i < numDumpHid; i++){
        for(int j = 0; j < numVis; j++){
            fprintf(weightFd, "%.5lf%s", weight[i*numVis+j], 
                ((j+1) % numVisPerLine == 0) ? "\n" : "\t");
        }
    }

    fclose(weightFd);
}

void RBM::getHProb(const double *v, double *ph, const int size){
    int i;
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, numHid, numVis,
                1, v, numVis, weight, numVis,
                0, ph, numHid);

    cblas_dger(CblasRowMajor, size, numHid,
               1.0, I(), 1, hbias, 1, ph, numHid);

    for(i = 0; i < size * numHid; i++){
        ph[i] = sigmoid(ph[i]);
    }
}

void RBM::getHSample(const double *ph, double *h, const int size){
    int i;

    for(i = 0; i < size * numHid; i++){
        h[i] = random_double(0, 1) < ph[i] ? 1 : 0; 
    }
}

void RBM::getVProb(const double *h, double *pv, const int size){
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, numVis, numHid,
                1, h, numHid, weight, numVis,
                0, pv, numVis);

    cblas_dger(CblasRowMajor, size, numVis,
               1.0, I(), 1, vbias, 1, pv, numVis);

    for(int i = 0; i < size * numVis; i++){
        pv[i] = sigmoid(pv[i]);
    }
}

void RBM::getVSample(const double *pv, double *v, const int size){
    int i;

    for(i = 0; i < size * numVis; i++){
        v[i] = random_double(0, 1) < pv[i] ? 1 : 0; 
    }
}

void RBM::gibbsSampleHVH(const double *hStart, double *h, double *ph, 
                      double *v, double *pv, const int step, const int size){
    cblas_dcopy(size * numHid, hStart, 1, h, 1);

    for(int i = 0; i < step; i++){
        getVProb(h, pv, size);
        getVSample(pv, v, size);
        getHProb(v, ph, size);
        getHSample(ph, h, size);
    }
}

void RBM::getFE(const double *v, double *FE, const int size){
    int i, j;
     
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, numHid, numVis,
                1, v, numVis, weight, numVis,
                0, temp, numHid);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, numHid, 1,
                1, I(), 1, hbias, 1,
                1, temp, numHid);

    for(i = 0; i < size * numHid; i++){
        temp[i] = log(exp(temp[i]) + 1.0);
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, 1, numHid,
                1, temp, numHid, I(), 1,
                0, FE, 1);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, 1, numVis,
                -1, v, numVis, vbias, 1,
                -1, FE, 1);
}

double RBM::getPL(double *v, const int size){
    static int filp_idx = 0;
    int i, j;
    double *FE, *FE_filp, *old_val;
    double PL = 0.0;

    FE = (double*)malloc(size * sizeof(double));
    FE_filp = (double*)malloc(size * sizeof(double));
    old_val = (double*)malloc(size * sizeof(double));

    getFE(v, FE, size);
    for(i = 0; i < size; i++){
        old_val[i] = v[i*numVis + filp_idx];
        v[i * numVis + filp_idx] = 1 - old_val[i];
    }
    getFE(v, FE_filp, size);
    for(i = 0; i < size; i++){
        v[i * numVis + filp_idx] = old_val[i];
    }

    for(i = 0; i < size; i++){
        PL += numVis * log(sigmoid(FE_filp[i] - FE[i]));
    }
    
    free(FE);
    free(FE_filp);
    free(old_val);

    filp_idx = (filp_idx + 1) % numVis;

    return PL / size;
}

void RBM::train(Dataset *data, double learningRate, int batchSize, int numEpoch){
    int numBatch = (data->getTrainingNumber()-1) / batchSize + 1;
    double *chainStart = NULL; //PCD

    for(int epoch = 0; epoch < numEpoch; epoch++){
        double cost = 0;
        time_t startTime = time(NULL);

        for(int k = 0; k < numBatch; k++){
#ifdef DEBUG
            if((k+1) % 10 == 0){
                printf("epoch %d batch %d\n", epoch + 1, k + 1);
            }
#endif
            v1 = data->getTrainingData(k*batchSize);
            int theBatchSize;

            if(k == (numBatch-1)){
                theBatchSize = data->getTrainingNumber() - batchSize * k;
            }else{
                theBatchSize = batchSize;
            }
            getHProb(v1, ph1, theBatchSize);
            getHSample(ph1, h1, theBatchSize);

            if(chainStart == NULL){ //PCD
                chainStart = h1;
            }
            //chain_start = H1;

            gibbsSampleHVH(chainStart, h2, ph2, v2, pv, 1, theBatchSize);

            // update weight
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        numHid, numVis, theBatchSize,
                        1.0, ph2, numHid, v2, numVis,
                        0, temp, numVis);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        numHid, numVis, theBatchSize,
                        1.0, ph1, numHid, v1, numVis,
                        -1, temp, numVis);

            cblas_daxpy(numHid*numVis, learningRate / theBatchSize,
                        temp, 1, weight, 1);

            // update vbias
            cblas_dcopy(numVis*theBatchSize, v1, 1, temp, 1);

            cblas_daxpy(numVis*theBatchSize, -1, v2, 1, temp, 1);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        numVis, 1, theBatchSize,
                        learningRate / theBatchSize, temp, numVis,
                        I(), 1, 1, vbias, 1);

            // update hbias
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        numHid, 1, theBatchSize,
                        1.0, ph2, numHid, I(), 1,
                        0, temp, 1);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        numHid, 1, theBatchSize,
                        1.0, ph1, numHid, I(), 1,
                        -1, temp, 1);

            cblas_daxpy(numHid, learningRate / theBatchSize,
                        temp, 1, hbias, 1);

            cost += getPL(v1, theBatchSize) / numBatch;
        }

        time_t endTime = time(NULL);
        printf("epoch %d cost: %.8lf\ttime: %.2lf min\n", epoch+1, cost, (double)(endTime - startTime) / 60);

        dumpWeight(100, 28);
    }
}
