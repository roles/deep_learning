#include "Dataset.h"
#include "TrainModel.h"
#include "DeepAutoEncoder.h"

using namespace std;

void testMNISTTraining(){
    MNISTDataset data;
    data.loadData();

    int sizes[] = {data.getFeatureNumber(), 1000, 500, 250, 2};

    DeepAutoEncoder dad(4, sizes);
    TrainModel model(dad);
    model.train(&data, 0.01, 10, 100);
}

void testMNISTFineTune(){
    MNISTDataset data;
    data.loadData();

    MultiLayerRBM* multirbm = new MultiLayerRBM("result/MNISTMultiLayerRBM_pretrain.dat");
    DeepAutoEncoder dad(*multirbm); 
    delete multirbm;
    TrainModel model(dad);
    model.train(&data, 0.01, 10, 100);
}

int main(){
    //testMNISTTraining();
    testMNISTFineTune();
}
