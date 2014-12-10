#include "MultiLayerRBM.h"
#include "mkl_cblas.h"
#include "Utility.h"
#include <cfloat>
#include <cstdio>

MultiLayerRBM::MultiLayerRBM(int numLayer, const int layersSize[]) :
    MultiLayerTrainComponent("MultiLayerRBM"), numLayer(numLayer), 
    layersToTrain(numLayer, true), persistent(true), AMSample(NULL)
{
    char weightFile[100], modelFile[100];
    for(int i = 0; i < numLayer; i++){
        layers[i] = new RBM(layersSize[i], layersSize[i+1]); 

        //sprintf(weightFile, "result/DBN_Layer%d_weight.txt", i+1);
        sprintf(modelFile, "result/DBN_Layer%d.dat", i+1);
        //layers[i]->setWeightFile(weightFile);
        layers[i]->setModelFile(modelFile);
    }
}

MultiLayerRBM::MultiLayerRBM(int numLayer, const vector<const char*> &layerModelFiles) :
    MultiLayerTrainComponent("MultiLayerRBM"), numLayer(numLayer), 
    layersToTrain(numLayer, true), persistent(true), AMSample(NULL)
{
    for(int i = 0; i < numLayer; i++){
        layers[i] = new RBM(layerModelFiles[i]);
    }
}

MultiLayerRBM::MultiLayerRBM(const char* file) :
    MultiLayerTrainComponent("MultiLayerRBM"), 
    persistent(true), AMSample(NULL)
{
    FILE* fd = fopen(file, "rb");
    if(fd == NULL){
        fprintf(stderr, "file not exist : %s\n", file);
        exit(1);
    }
    fread(&numLayer, sizeof(int), 1, fd);
    for(int i = 0; i < numLayer; i++){
        layers[i] = new RBM(fd);
    }
    fclose(fd);
    layersToTrain = vector<bool>(numLayer, true);
}

void MultiLayerRBM::saveModel(FILE* fd){
    fwrite(&numLayer, sizeof(int), 1, fd);
    for(int i = 0; i < numLayer; i++){
        layers[i]->saveModel(fd);
    }
}

MultiLayerRBM::~MultiLayerRBM(){
    for(int i = 0; i < numLayer; i++){
        delete layers[i];
    }
    delete[] AMSample;
}

void MultiLayerRBM::setPersistent(bool p){
    for(int i = 0; i < numLayer; i++)
        layers[i]->setPersistent(p);
}

TrainComponent& MultiLayerRBM::getLayer(int i){
    return *layers[i];
}

void MultiLayerRBM::toMLP(MLP* mlp, int lastNumOut){
    double *transWeight = new double[maxUnit*maxUnit];
    mlp->setLayerNumber(numLayer);
    for(int i = 0; i < numLayer; i++){
        layers[i]->getWeightTrans(transWeight);
        mlp->setLayer(i, new SigmoidLayer(layers[i]->numVis, layers[i]->numHid, transWeight, layers[i]->hbias));
    }
    mlp->addLayer(new Logistic(layers[numLayer-1]->numHid, lastNumOut));
    delete[] transWeight;
}

void MultiLayerRBM::loadLayer(int i, const char* file){
    delete layers[i];
    layers[i] = new RBM(file);
}

void MultiLayerRBM::addLayer(int numLayerHid){
    char modelFile[100];
    layers[numLayer] = new RBM(layers[numLayer-1]->numHid, numLayerHid);
    sprintf(modelFile, "result/DBN_Layer%d.dat", numLayer+1);
    layers[numLayer]->setModelFile(modelFile);
    layersToTrain.push_back(true);
    numLayer++;
}

void MultiLayerRBM::activationMaxization(int layerIdx, int unitNum, double avgNorm, int nepoch,
        const char amSampleFile[])
{
    int topHiddenNum = layers[layerIdx]->numHid;
    int bottomVisibleNum = layers[0]->numVis;
    FILE* fd = fopen(amSampleFile, "w+");

    if(AMSample == NULL){
        AMSample = new double[topHiddenNum*bottomVisibleNum];
    }
    for(int i = 0; i < topHiddenNum*bottomVisibleNum; i++){
        AMSample[i] = random_double(0, 1);
    }
    for(int i = 0; i < unitNum; i++){
        double *unitSample = AMSample + i*bottomVisibleNum;
        time_t startTime = time(NULL);
        double maxval = maximizeUnit(layerIdx, i, unitSample, avgNorm, nepoch);
        time_t endTime = time(NULL);

        printf("layer %d unit %d maximum : %.8lf\t time : %.2lfmin\n",
                layerIdx+1, i+1, maxval , (double)(endTime - startTime) / 60);
        fflush(stdout);
        layers[0]->dumpSample(fd, unitSample, 1);
        fflush(fd);
    }

    fclose(fd);
}

double MultiLayerRBM::maximizeUnit(int layerIdx, int unitIdx, 
        double* unitSample, double avgNorm, int nepoch)
{
    int topHiddenNum = layers[layerIdx]->numHid;
    int bottomVisibleNum = layers[0]->numVis;
    int k = 0;

    // average norm
    double curNorm = squareNorm(unitSample, bottomVisibleNum, 1);
    cblas_dscal(bottomVisibleNum, avgNorm / curNorm, unitSample, 1);

    double curval;

    while(k++ < nepoch){
        
        // forward
        for(int i = 0; i <= layerIdx; i++){
            if(i == 0){
                layers[i]->setInput(unitSample);
            }else{
                layers[i]->setInput(layers[i-1]->getOutput());
            }
            layers[i]->runBatch(1);
        }
        curval = layers[layerIdx]->getOutput()[unitIdx];
        //printf("unit index %d epoch %d current maximal : %.8lf\n", unitIdx+1, k, curval);

        // back-propagate
        for(int i = layerIdx; i >= 0; i--){
            if(i == layerIdx){
                layers[i]->getAMDelta(unitIdx, NULL);
            }else{
                layers[i]->getAMDelta(-1, layers[i+1]->AMdelta);
            }
        }

        // 自适应learning rate
        double lr = 0.01 * cblas_dasum(bottomVisibleNum, unitSample, 1) /
                    cblas_dasum(bottomVisibleNum, layers[0]->AMdelta, 1);

        // update sample
        cblas_daxpy(bottomVisibleNum, lr, 
                    layers[0]->AMdelta, 1, 
                    unitSample, 1);

        // average norm
        curNorm = squareNorm(unitSample, bottomVisibleNum, 1);
        cblas_dscal(bottomVisibleNum, avgNorm / curNorm, unitSample, 1);
    }
    return curval;
}
