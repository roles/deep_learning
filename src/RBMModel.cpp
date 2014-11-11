#include "RBM.h"

void testMNIST(){
    MNISTDataset mnist;
    mnist.loadData();

    RBM rbm(mnist.getFeatureNumber(), 500); //500个隐藏结点
    //rbm.setModelFile("result/RBMModel.dat");
    rbm.setWeightFile("result/mnist_rbm_weight.txt"); //设置导出权重的文件

    TrainModel RBMModel(rbm);
    RBMModel.train(&mnist, 0.1, 20, 5);
}

void testMNISTLoading(){
    MNISTDataset mnist;
    mnist.loadData();

    RBM rbm("result/RBMModel.dat"); //500个隐藏结点
    rbm.setWeightFile("result/mnist_rbm_weight.txt"); //设置导出权重的文件
    rbm.dumpWeight(100, 28);
}

int main(){
    srand(1234);
    testMNIST();
    //testMNISTLoading();
}
