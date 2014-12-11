#include "Dataset.h"
#include "DeepClassRBM.h"

void testMNISTDeepClassRBMTraining(){
    MNISTDataset data;
    data.loadData();
    
    MultiLayerRBM* multirbm = new MultiLayerRBM("result/MNISTMultiLayerRBM_1000_1000_1000_0.01.dat");
    multirbm->setLayerToTrain(0, false);
    multirbm->setLayerToTrain(1, false);
    multirbm->setLayerToTrain(2, false);

    ClassRBM* classrbm = new ClassRBM(multirbm->getTopOutputNumber(), 500, data.getLabelNumber());
    DeepClassRBM deepClassrbm(multirbm, classrbm);

    MultiLayerTrainModel model(deepClassrbm);
    model.train(&data, 0.01, 10, 1000);
}

void testTCGADeepClassRBMTraining(){
    TCGADataset data;
    data.loadData();
    
    MultiLayerRBM* multirbm = new MultiLayerRBM("result/TCGAMultiLayerRBM_2000_0.01_13epoch_2000_0.01_3epoch.dat");
    //MultiLayerRBM* multirbm = new MultiLayerRBM("result/TCGAMultiLayerRBM_2000_0.01_13epoch_2000_0.01_3epoch_2000_0.01_16epoch.dat");
    multirbm->setLayerToTrain(0, false);
    multirbm->setLayerToTrain(1, false);
    //multirbm->setLayerToTrain(2, false);

    ClassRBM* classrbm = new ClassRBM(multirbm->getTopOutputNumber(), 1000, data.getLabelNumber());
    DeepClassRBM deepClassrbm(multirbm, classrbm);

    MultiLayerTrainModel model(deepClassrbm);
    model.train(&data, 0.01, 1, 1000);
}

int main(){
    testTCGADeepClassRBMTraining();
    return 0;
}
