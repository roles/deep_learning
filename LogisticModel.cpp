#include "Logistic.h"

void testICA(){
    TrivialDataset data;
    data.loadData("../data/minist1_lcn_mlp.bin", "../data/minist1_lcn_label.bin");

    Logistic logi(data.getFeatureNumber(), data.getLabelNumber());
    TrainModel logisticModel(logi);
    logisticModel.train(&data, 0.13, 500, 1000);
}

void testMNIST(){
    MNISTDataset mnist;
    mnist.loadData();

    Logistic logi(mnist.getFeatureNumber(), mnist.getLabelNumber());
    TrainModel logisticModel(logi);
    logisticModel.train(&mnist, 0.13, 500, 1000);
}

int main(){
    //testMNIST();
    testICA();
}
