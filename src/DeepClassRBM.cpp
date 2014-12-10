#include "ClassRBM.h"
#include "MultiLayerRBM.h"
#include "Dataset.h"

class DeepClassRBM : public MultiLayerTrainComponent {
    public:
        DeepClassRBM(MultiLayerRBM*, ClassRBM*);
        ~DeepClassRBM();
        TrainComponent& getLayer(int);
        int getLayerNumber();
        bool getLayerToTrain(int);
        void saveModel(FILE* fd);
    private:
        MultiLayerRBM* multirbm;
        ClassRBM* classrbm;
        int numLayer;
};

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

void testDeepClassRBMTraining(){
    MNISTDataset data;
    data.loadData();
    
    MultiLayerRBM* multirbm = new MultiLayerRBM("MNISTMultiLayerRBM_1000_1000_1000_0.01.dat");
    multirbm->setLayerToTrain(0, false);
    multirbm->setLayerToTrain(1, false);
    multirbm->setLayerToTrain(2, false);

    ClassRBM* classrbm = new ClassRBM(multirbm->getTopOutputNumber(), 500, data.getLabelNumber());
    DeepClassRBM deepClassrbm(multirbm, classrbm);

    MultiLayerTrainModel model(deepClassrbm);
    model.train(&data, 0.01, 10, 1000);
}

int main(){
    testDeepClassRBMTraining();
    return 0;
}
