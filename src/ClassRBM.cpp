#include "ClassRBM.h"
#include "Config.h"
#include "Utility.h"
#include "mkl_cblas.h"
#include <cstring>
#include <cfloat>

typedef double TempBuffer[maxUnit*maxUnit];
static TempBuffer temp, temp2, temp3, temp4;


ClassRBM::ClassRBM(int numVis, int numHid, int numLabel) :
    TrainComponent(Supervise, "ClassRBM"),
    numVis(numVis), numHid(numHid), numLabel(numLabel),
    h(NULL), ph(NULL), py(NULL), phk(NULL), alpha(0)
{
    W = new double[numVis*numHid];
    U = new double[numLabel*numHid];
    b = new double[numVis];
    c = new double[numHid];
    d = new double[numLabel];

    initializeWeightSigmoid(W, numVis, numHid);
    initializeWeightSigmoid(U, numLabel, numHid);
    memset(b, 0, numVis*sizeof(double));
    memset(c, 0, numHid*sizeof(double));
    memset(d, 0, numLabel*sizeof(double));
}

ClassRBM::~ClassRBM(){
    delete[] W;
    delete[] U;
    delete[] b;
    delete[] c;
    delete[] d;

    delete[] h;
    delete[] ph;
    delete[] py;
    delete[] phk;

    delete[] xGen;
    delete[] yGen;
}

void ClassRBM::beforeTraining(int size){
    initBuffer(size);
}

void ClassRBM::afterTraining(int size){

}

void ClassRBM::trainBatch(int size){
    forward(size);
    update(size);
}

void ClassRBM::runBatch(int size){
    getYProb(x, py, size);
}

void ClassRBM::forward(int size){

    getHProb(x, y, ph, size);
    getYProb(x, py, size);

    if(alpha != 0){
        binomial(ph, h, size*numHid);
        getYFromH(h, yGen, size);
        getXFromH(h, xGen, size);
    }

    double *ty = temp;
    memset(ty, 0, sizeof(double)*size*numLabel);

    for(int i = 0; i < numLabel; i++){

        for(int j = 0; j < size; j++){  //set label
            ty[j*numLabel+i] = 1.0;
        }

        getHProb(x, ty, &phk[i*size*numHid], size);

        for(int j = 0; j < size; j++){
            cblas_dscal(numHid, py[j*numLabel+i] , 
                    &phk[i*size*numHid+j*numHid], 1);
        }

        for(int j = 0; j < size; j++){  //reset label
            ty[j*numLabel+i] = 0;
        }
    }
}

void ClassRBM::getXFromH(double *h, double *x, int size){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, numVis, numHid,
                1.0, h, numHid, W, numVis,
                0, x, numVis);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, numVis, 1,
                1.0, I(), 1, b, 1,
                1.0, x, numVis);

    for(int i = 0; i < size*numVis; i++)
        x[i] = sigmoid(x[i]);

    binomial(x, x, size*numVis);
}

void ClassRBM::getYFromH(double *h, double *y, int size){
    double *hU = temp;
    double *tpy = temp2;
    double *ty = temp3;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, numLabel, numHid,
                1.0, h, numHid, U, numLabel,
                0, hU, numLabel);
    
    for(int i = 0; i < numLabel; i++){

        for(int j = 0; j < size; j++){  //set label
            ty[j*numLabel+i] = 1.0;
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    size, 1, numLabel,
                    1.0, hU, numLabel, ty, 1,
                    0, tpy, 1);

        for(int j = 0; j < size; j++){
            y[j*numLabel+i] = tpy[j];
        }

        for(int j = 0; j < size; j++){  //reset label
            ty[j*numLabel+i] = 0;
        }
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, numLabel, 1,
                1.0, I(), 1, d, 1,
                1.0, y, numLabel);

    for(int i = 0; i < size; i++){
        softmax(y+i*numLabel, numLabel);
    }

    for(int i = 0; i < size; i++){
        multiNormial(y+i*numLabel, y+i*numLabel, numLabel);
    }
}

void ClassRBM::updateW(int size){
    double *deltaSum = temp,
           *delta = temp2;

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            numHid, numVis, size,
            1.0, ph, numHid, x, numVis,
            0, delta, numVis);

    memset(deltaSum, 0, sizeof(double)*numVis*numHid);

    for(int i = 0; i < numLabel; i++){
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numHid, numVis, size,
                1.0, &phk[i*size*numHid], numHid, x, numVis,
                1.0, deltaSum, numVis);
    }

    cblas_daxpy(numVis*numHid, -1.0, deltaSum, 1, delta, 1);

    cblas_daxpy(numVis*numHid, learningRate / size, delta, 1, W, 1);
}

void ClassRBM::updateU(int size){
    double *deltaSum = temp,
           *delta = temp2,
           *ty = temp3;
    memset(ty, 0, sizeof(double)*size*numLabel);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            numHid, numLabel, size,
            1.0, ph, numHid, y, numLabel,
            0, delta, numLabel);

    memset(deltaSum, 0, sizeof(double)*numLabel*numHid);

    for(int i = 0; i < numLabel; i++){

        for(int j = 0; j < size; j++){  //set label
            ty[j*numLabel+i] = 1.0;
        }

        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numHid, numLabel, size,
                1.0, &phk[i*size*numHid], numHid, ty, numLabel,
                1.0, deltaSum, numLabel);

        for(int j = 0; j < size; j++){  //reset label
            ty[j*numLabel+i] = 0;
        }
    }

    cblas_daxpy(numLabel*numHid, -1.0, deltaSum, 1, delta, 1);

    cblas_daxpy(numLabel*numHid, learningRate / size, delta, 1, U, 1);
}

void ClassRBM::updateHbias(int size){
    double *deltaSum = temp,
           *delta = temp2;

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            numHid, 1, size,
            1.0, ph, numHid, I(), 1,
            0, delta, 1);

    memset(deltaSum, 0, sizeof(double)*numVis*numHid);

    for(int i = 0; i < numLabel; i++){
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                numHid, 1, size,
                1.0, &phk[i*size*numHid], numHid, I(), 1,
                1.0, deltaSum, 1);
    }

    cblas_daxpy(numHid, -1.0, deltaSum, 1, delta, 1);

    cblas_daxpy(numHid, learningRate / size, delta, 1, c, 1);
}

void ClassRBM::updateYBias(int size){
    double *delta = temp,
           *deltaSum = temp2;

    cblas_dcopy(size*numLabel, y, 1, deltaSum, 1);

    cblas_daxpy(size*numLabel, -1.0, py, 1, deltaSum, 1);

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            numLabel, 1, size,
            1.0, deltaSum, numLabel, I(), 1,
            0, delta, 1);

    cblas_daxpy(numLabel, learningRate / size, delta, 1, d, 1);
}

void ClassRBM::updateXBias(int size){

}

void ClassRBM::update(int size){
    updateW(size); 
    updateU(size);
    updateHbias(size);
    updateYBias(size);
    if(alpha != 0)
        updateXBias(size);
}

void ClassRBM::initBuffer(int size){
    if(h == NULL)
        h = new double[size*numHid];
    if(ph == NULL)
        ph = new double[size*numHid];
    if(py == NULL)
        py = new double[size*numLabel];
    if(phk == NULL)
        phk = new double[size*numHid*numLabel];
    if(alpha != 0){
        if(yGen == NULL)
            yGen = new double[size*numLabel];
        if(xGen == NULL)
            xGen = new double[size*numVis];
    }
}

void ClassRBM::getHProb(double *x, double *y, double *ph, int size){

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            size, numHid, numVis,
            1.0, x, numVis, W, numVis,
            0, ph, numHid);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            size, numHid, numLabel,
            1.0, y, numLabel, U, numLabel,
            1.0, ph, numHid);

    cblas_dger(CblasRowMajor, size, numHid,
            1.0, I(), 1, c, 1, ph, numHid);

    for(int i = 0; i < numHid * size; i++){
        ph[i] = sigmoid(ph[i]);
    }
}

void ClassRBM::getYProb(double *x, double *py, int size){
    double *ty = temp, *preAddU = temp2, 
           *softp = temp3, *res = temp4;

    memset(ty, 0, sizeof(double)*size*numLabel);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            size, numHid, numVis,
            1.0, x, numVis, W, numVis,
            0, preAddU, numHid);

    cblas_dger(CblasRowMajor, size, numHid,
            1.0, I(), 1, c, 1, preAddU, numHid);

    for(int i = 0; i < numLabel; i++){
        for(int j = 0; j < size; j++){  //set label
            ty[j*numLabel+i] = 1.0;
        }

        cblas_dcopy(size * numHid, preAddU, 1, softp, 1);

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, numHid, numLabel,
                1.0, ty, numLabel, U, numLabel,
                1.0, softp, numHid);

        for(int j = 0; j < size * numHid; j++){
            if(softp[j] < 30){
                softp[j] = softplus(softp[j]);
            } 
        }

        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, 1, numHid,
                1.0, softp, numHid, I(), 1,
                0, res, 1);

        for(int j = 0; j < size; j++){
            //py[j*numLabel+i] = exp(d[i] + res[j]);

            py[j*numLabel+i] = d[i] + res[j];

            if(isnan(py[j*numLabel+i]) || isinf(py[j*numLabel+i])){
                printf("nan inf occur %.10lf\n", d[i]+res[j]);
                exit(1);
            }
        }

        for(int j = 0; j < size; j++){  //reset label
            ty[j*numLabel+i] = 0;
        }
    }

    // substract the maximum to avoid exp inf
    for(int i = 0; i < size; i++){
        /*
        double maxval = -DBL_MAX;
        for(int j = 0; j < numLabel; j++){
            if(py[i*numLabel+j] > maxval)
                maxval = py[i*numLabel+j];
        }
        for(int j = 0; j < numLabel; j++){
            py[i*numLabel+j] = exp(py[i*numLabel+j] - maxval);
        }
        */
        softmax(py+i*numLabel, numLabel);
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            size, 1, numLabel,
            1.0, py, numLabel, I(), 1,
            0, res, 1);

    for(int i = 0; i < size; i++){
        for(int j = 0; j < numLabel; j++){
            py[i*numLabel+j] /= res[i];
        }
    }
}

void ClassRBM::saveModel(FILE* fd){
    fwrite(&numVis, sizeof(int), 1, fd);
    fwrite(&numHid, sizeof(int), 1, fd);
    fwrite(&numLabel, sizeof(int), 1, fd);
    fwrite(W, sizeof(double), numVis*numHid, fd);
    fwrite(U, sizeof(double), numLabel*numHid, fd);
    fwrite(b, sizeof(double), numVis, fd);
    fwrite(c, sizeof(double), numHid, fd);
    fwrite(d, sizeof(double), numLabel, fd);
}

