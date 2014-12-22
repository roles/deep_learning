#include "RBM.h"
#include "mkl_cblas.h"

static double temp[maxUnit*maxUnit], temp2[maxUnit*maxUnit];

RBM::RBM(int numVis, int numHid)
    : numVis(numVis), numHid(numHid), chainStart(NULL), persistent(true),
      v1(NULL), v2(NULL), pv(NULL),
      h1(NULL), h2(NULL), ph1(NULL), ph2(NULL), AMdelta(NULL),
      weightFile(NULL), UnsuperviseTrainComponent("RBM"), sparsity(false)
{
    weight = new double[numVis*numHid];
    initializeWeightSigmoid(weight, numVis, numHid);

    vbias = new double[numVis];
    hbias = new double[numHid];
    memset(vbias, 0, numVis*sizeof(double));
    memset(hbias, 0, numHid*sizeof(double));
}

void RBM::loadModel(FILE* fd)
{
    fread(&numVis, sizeof(int), 1, fd);
    fread(&numHid, sizeof(int), 1, fd);
    printf("numVis : %d\nnumHid : %d\n", numVis, numHid);

    weight = new double[numVis*numHid];
    vbias = new double[numVis];
    hbias = new double[numHid];

    fread(weight, sizeof(double), numVis*numHid, fd);
    fread(vbias, sizeof(double), numVis, fd);
    fread(hbias, sizeof(double), numHid, fd);
}

RBM::RBM(const char *modelFile) : chainStart(NULL), persistent(true),
      v1(NULL), v2(NULL), pv(NULL),
      h1(NULL), h2(NULL), ph1(NULL), ph2(NULL), AMdelta(NULL),
      weightFile(NULL), UnsuperviseTrainComponent("RBM"), sparsity(false)
{
    FILE* fd = fopen(modelFile, "rb");

    if(fd == NULL){
        fprintf(stderr, "cannot open file %s\n", modelFile);
        exit(1);
    }
    loadModel(fd);
    fclose(fd);
}

RBM::RBM(FILE* fd) : chainStart(NULL), persistent(true),
      v1(NULL), v2(NULL), pv(NULL),
      h1(NULL), h2(NULL), ph1(NULL), ph2(NULL), AMdelta(NULL),
      weightFile(NULL), UnsuperviseTrainComponent("RBM"), sparsity(false)
{
    loadModel(fd);
}

RBM::~RBM(){
    delete[] weight;
    delete[] vbias;
    delete[] hbias;
    delete[] weightFile;
    freeBuffer();

    delete[] AMdelta;
}

void RBM::allocateBuffer(int size){
    if(v2 == NULL) v2 = new double[size*numVis];
    if(h1 == NULL) h1 = new double[size*numHid];
    if(h2 == NULL) h2 = new double[size*numHid];
    if(pv == NULL) pv = new double[size*numVis];
    if(ph1 == NULL) ph1 = new double[size*numHid];
    if(ph2 == NULL) ph2 = new double[size*numHid];
    if(AMdelta == NULL) AMdelta = new double[numVis];
}

void RBM::freeBuffer(){
    delete[] v2;
    delete[] h1;
    delete[] h2;
    delete[] pv;
    delete[] ph1;
    delete[] ph2;
    v2 = h1 = h2 = pv = ph1 = ph2 = NULL;
}

void RBM::beforeTraining(int size){
    allocateBuffer(size);
}

void RBM::afterTraining(int size){
    TrainComponent::afterTraining(size);
    freeBuffer();
}

void RBM::trainBatch(int size){
    runChain(size, 1);
    updateWeight(size);
    updateBias(size);
}

/**
 * @brief  这个函数执行完之后可以通过getOutput获取结果
 *
 * @param size
 */
void RBM::runBatch(int size){
    allocateBuffer(size);
    runChain(size, 1);
}

void RBM::runChain(int size, int step){
    getHProb(v1, ph1, size);
    getHSample(ph1, h1, size);
    if(persistent){
        if(chainStart == NULL){ //PCD
            chainStart = h2;
            memset(h2, 0, sizeof(double)*numHid*size);
        }
        gibbsSampleHVH(h2, h2, ph2, v2, pv, step, size);
        //gibbsSampleHVH(chainStart, h2, ph2, v2, pv, step, size);
        //chainStart = h2;
    }else{
        gibbsSampleHVH(h1, h2, ph2, v2, pv, step, size);
    }

    if(sparsity){
        double cq = cblas_dasum(numHid * size, ph1, 1) / (numHid * size);
        q = numda * q + (1 - numda) * cq;
    }
}

void RBM::setLearningRate(double lr){
    learningRate = lr;
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

void RBM::gibbsSampleHVH(double *hStart, double *h, double *ph, 
                      double *v, double *pv, const int step, const int size){
    if(h != hStart)
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

/*
 * cross-entropy
 */
double RBM::getReconstructCost(double *v, double *pv, int size){
    double res;

    for(int i = 0; i < numVis * size; i++){
        temp[i] = v[i] >= 1.0 ? 1.0 : v[i];
        temp2[i] = log(pv[i] + 1e-10);
    }
    res = cblas_ddot(size * numVis, temp, 1, temp2, 1);

    for(int i = 0; i < numVis * size; i++){
        temp[i] = 1.0 - v[i] >= 1.0 ? 1.0 : 1.0 - v[i];
        temp2[i] = log(1.0 - pv[i] + 1e-10);
    }

    res += cblas_ddot(size * numVis, temp, 1, temp2, 1);
    return res / size;
}

void RBM::updateWeight(int size){
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            numHid, numVis, size,
            1.0, ph2, numHid, v2, numVis,
            0, temp, numVis);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            numHid, numVis, size,
            1.0, ph1, numHid, v1, numVis,
            -1, temp, numVis);

    //cblas_dscal(numVis*numHid, 1.0 - 2.0 * L2Reg * learningRate , weight, 1);

    cblas_daxpy(numHid*numVis, learningRate / size,
            temp, 1, weight, 1);

    if(sparsity){
        double* hderiv = temp;
        double* delta = temp2;
        for(int i = 0; i < size * numHid; i++){
            hderiv[i] = get_sigmoid_derivative(ph1[i]);
        }

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numHid, numVis, size,
                1.0, hderiv, numHid, v1, numVis,
                0, delta, numVis);

        double param = slr * (q - p) / 
                       ( (1-q+1e-10) * (q+1e-10) * numHid * size);
        cblas_daxpy(numHid*numVis, param,
                delta, 1, weight, 1);
    }
}

void RBM::updateBias(int size){
    // update vbias
    cblas_dcopy(numVis*size, v1, 1, temp, 1);

    cblas_daxpy(numVis*size, -1, v2, 1, temp, 1);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            numVis, 1, size,
            learningRate / size, temp, numVis,
            I(), 1, 1, vbias, 1);

    // update hbias
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            numHid, 1, size,
            1.0, ph2, numHid, I(), 1,
            0, temp, 1);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            numHid, 1, size,
            1.0, ph1, numHid, I(), 1,
            -1, temp, 1);

    cblas_daxpy(numHid, learningRate / size,
            temp, 1, hbias, 1);

    if(sparsity){
        double* hderiv = temp;
        double* delta = temp2;
        for(int i = 0; i < size * numHid; i++){
            hderiv[i] = get_sigmoid_derivative(ph1[i]);
        }

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numHid, 1, size,
                1.0, hderiv, numHid, I(), 1,
                0, delta, 1);

        double param = slr * (q - p) / 
                       ( (1-q+1e-10) * (q+1e-10) * numHid * size);
        cblas_daxpy(numHid, param, delta, 1, hbias, 1);
    }
}

double RBM::getTrainingCost(int size, int numBatch){
    if(persistent){
        return getPL(v1, size);
    }else{
        return getReconstructCost(v1, pv, size);
    }
}

void RBM::operationPerEpoch(){
    //dumpWeight(100, 28);
    if(sparsity)
        printf("\t===> sparsity : %.8lf\n", q);
}

void RBM::saveModel(FILE *fd){
    fwrite(&numVis, sizeof(int), 1, fd);
    fwrite(&numHid, sizeof(int), 1, fd);
    fwrite(weight, sizeof(double), numVis*numHid, fd);
    fwrite(vbias, sizeof(double), numVis, fd);
    fwrite(hbias, sizeof(double), numHid, fd);
}

void RBM::generateSample(const char* sampleFile, double* v, int count){
    FILE* fd = fopen(sampleFile, "w+"); 
    int chainStep = 10, generateCount = 10;
    allocateBuffer(count);

    dumpSample(fd, v, count);
    getHProb(v, ph2, count);
    getHSample(ph2, h2, count);
    for(int i = 0; i < generateCount-1; i++){
        gibbsSampleHVH(h2, h2, ph2, v2, pv, chainStep, count);
        dumpSample(fd, pv, count);
    } 
    freeBuffer();
    fclose(fd);
}

void RBM::dumpSample(FILE* fd, double *v, int count){
    int numVisPerLine = 28;
    for(int i = 0; i < count; i++){
        for(int j = 0; j < numVis; j++){
            fprintf(fd, "%.5lf%s", v[i*numVis+j], 
                ((j+1) % numVisPerLine == 0) ? "\n" : "\t");
        }
    }
}

void RBM::dumpSampleBinary(FILE* fd, double *v, int count){
    fwrite(v, sizeof(double), count*numVis, fd); 
}

void RBM::getWeightTrans(double *transWeight){
    for(int i = 0; i < numHid; i++)
        for(int j = 0; j < numVis; j++)
            transWeight[j*numHid+i] = weight[i*numVis+j];
}

/**
 * @brief  计算最大化激励
 *
 * @param unitIdx 如果lastAMdelta为NULL，则表示最大化该层的第unitIdx个隐藏结点
 *                否则unitIdx为-1
 * @param lastAMdelta 上层计算的梯度
 */
void RBM::getAMDelta(int unitIdx, double *lastAMdelta){
    if(unitIdx == -1){
        for(int i = 0; i < numHid; i++){
            temp[i] = get_sigmoid_derivative(ph1[i]) * lastAMdelta[i];
        }
    }else{
        for(int i = 0; i < numHid; i++){
            if(i == unitIdx){
                temp[i] = get_sigmoid_derivative(ph1[i]);
            }else{
                temp[i] = 0;
            }
        }
    }
    
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numVis, 1, numHid,
                1, weight, numVis, temp, 1,
                0, AMdelta, 1);
}
