#include "RBM.h"

using namespace std;

RBMBuffer V2, H1, H2, Ph1, Ph2, Pv;

/**
 * @brief  初始化全局计算buffer
 */
void initGlobalData(){

}

/**
 * @brief  
 */
void freeGlobalData(){

}

int main(){
    initGlobalData();

    MNISTDataset mnist;
    mnist.loadData();

    RBM rbm(mnist.getFeatureNumber(), 500); //500个隐藏结点
    rbm.setBuffer(V2, H1, H2, Pv, Ph1, Ph2); //设置计算缓冲区
    rbm.setWeightFile("mnist_rbm_weight.txt"); //设置导出权重的文件

    rbm.train(&mnist);

    freeGlobalData();
}
