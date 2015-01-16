#include "Logistic.h"

void testICA(){
    TrivialDataset data;
    data.loadData("../data/minist1_lcn_mlp.bin", "../data/minist1_lcn_label.bin");

    Logistic logi(data.getFeatureNumber(), data.getLabelNumber());
    TrainModel logisticModel(logi);
    logisticModel.train(&data, 0.13, 500, 1000);
}

void testMNISTTraining(){
    MNISTDataset mnist;
    mnist.loadData();

    Logistic logi(mnist.getFeatureNumber(), mnist.getLabelNumber());
    logi.setModelFile("result/LogisticModel.dat");
    TrainModel logisticModel(logi);
    logisticModel.train(&mnist, 0.13, 500, 100);
}

void testMNISTDataLoading(){
    MNISTDataset mnist;
    mnist.loadData();

    Logistic logi("result/LogisticModel.dat");
    TrainModel logisticModel(logi);
    printf("validate error : %.8lf%%\n", 100.0 * logisticModel.getValidError(&mnist, 500));
}

void testTCGATraining(){
    TCGADataset tcga;
    tcga.loadData();

    Logistic logi(tcga.getFeatureNumber(), tcga.getLabelNumber());
    logi.setModelFile("result/TCGALogisticModel.dat");
    TrainModel logisticModel(logi);
    logisticModel.train(&tcga, 0.01, 1, 1000, 5, -0.005);
}

int main(){
    //testMNISTTraining();
    //testMNISTDataLoading();
    //testICA();

    testTCGATraining();
}
