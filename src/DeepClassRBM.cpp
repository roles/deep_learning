#include "DeepClassRBM.h"

DeepClassRBM::DeepClassRBM(MultiLayerRBM* multirbm, ClassRBM* classrbm) :
    MultiLayerTrainComponent("DeepClassRBM"),
    multirbm(multirbm), classrbm(classrbm)
{
    numLayer = multirbm->getLayerNumber() + 1;
}

DeepClassRBM::~DeepClassRBM(){
    delete multirbm;
    delete classrbm;
}

int DeepClassRBM::getLayerNumber(){
    return numLayer;
}

TrainComponent& DeepClassRBM::getLayer(int layerIdx){
    if(layerIdx == numLayer-1){
        return *classrbm;
    }else{
        return multirbm->getLayer(layerIdx);
    }
}

bool DeepClassRBM::getLayerToTrain(int layerIdx){ 
    if(layerIdx == numLayer-1){
        return true;
    }else{
        return multirbm->getLayerToTrain(layerIdx);
    }
}

void DeepClassRBM::saveModel(FILE* fd){

}

