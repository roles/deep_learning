#include "MultiLayerTrainComponent.h"
#include "Dataset.h"

MultiLayerTrainComponent::MultiLayerTrainComponent(const char* name) : IModel(name) { }

MultiLayerTrainComponent::~MultiLayerTrainComponent(){ }

int MultiLayerTrainComponent::getBottomInputNumber(){
    return getLayer(0).getInputNumber(); 
}

int MultiLayerTrainComponent::getTopOutputNumber(){
    return getLayer(getLayerNumber()-1).getOutputNumber();
}
