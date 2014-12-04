#include "MultiLayerRBM.h"

MultiLayerRBM::MultiLayerRBM(int numLayer, const int layersSize[]) :
    MultiLayerTrainComponent("MultiLayerRBM"), numLayer(numLayer), 
    layersToTrain(numLayer, true), persistent(true)
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
    layersToTrain(numLayer, true), persistent(true)
{
    for(int i = 0; i < numLayer; i++){
        layers[i] = new RBM(layerModelFiles[i]);
    }
}

MultiLayerRBM::MultiLayerRBM(const char* file) :
    MultiLayerTrainComponent("MultiLayerRBM"), 
    persistent(true)
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
