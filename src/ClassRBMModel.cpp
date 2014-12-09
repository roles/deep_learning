#include "ClassRBM.h"
#include "Dataset.h"
#include "TrainModel.h"
#include <cmath>

void testTrainMNIST(){
    MNISTDataset mnist;
    mnist.loadData();

    ClassRBM classrbm(mnist.getFeatureNumber(), 500, mnist.getLabelNumber()); //500个隐藏结点

    TrainModel ClassRBMModel(classrbm);
    ClassRBMModel.train(&mnist, 0.05, 20, 1);
}

void testTCGATraining(){
    TCGADataset data;
    data.loadData();

    ClassRBM classrbm(data.getFeatureNumber(), 2000, data.getLabelNumber()); //500个隐藏结点

    TrainModel ClassRBMModel(classrbm);
    ClassRBMModel.train(&data, 0.01, 1, 1000, 20);
}

void test20NewsGroup(){
    SVMDataset data;
    data.loadData("../data/20newsgroup_train.txt", "../data/20newsgroup_valid.txt");

    ClassRBM classrbm(data.getFeatureNumber(), 2000, data.getLabelNumber()); //500个隐藏结点

    TrainModel ClassRBMModel(classrbm);
    ClassRBMModel.train(&data, 0.01, 10, 1, 20);
}

int main(){
    //testTrainMNIST();
    //testTCGATraining();
    test20NewsGroup();
    return 0;
}
