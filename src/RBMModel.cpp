#include "RBM.h"

void testMNIST(){
    MNISTDataset mnist;
    mnist.loadData();

    RBM rbm(mnist.getFeatureNumber(), 500); //500个隐藏结点
    //rbm.setModelFile("result/RBMModel.dat");
    rbm.setWeightFile("result/mnist_rbm_weight.txt"); //设置导出权重的文件

    TrainModel RBMModel(rbm);
    RBMModel.train(&mnist, 0.1, 20, 15);
}

void testMNISTLoading(){
    MNISTDataset mnist;
    mnist.loadData();

    RBM rbm("result/RBMModel.dat"); //500个隐藏结点
    rbm.setWeightFile("result/mnist_rbm_weight.txt"); //设置导出权重的文件
    rbm.dumpWeight(100, 28);
}

void testMNISTDumpSample(){
    MNISTDataset mnist;
    mnist.loadData();

    //RBM rbm("result/DBN_Layer1.dat");
    RBM rbm("result/RBMModel.dat");
    double *sample = mnist.getValidateData(0);
    int sampleCount = 20;

    rbm.generateSample("result/rbm_sample.txt", sample, sampleCount);
}

int main(){
    srand(1234);
    //testMNIST();
    //testMNISTLoading();
    testMNISTDumpSample();
}
