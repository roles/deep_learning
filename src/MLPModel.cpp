#include "Dataset.h"
#include "MLP.h"

void testWFICA(){
    TrivialDataset data;
    data.loadData("../data/minist1_lcn_mlp.bin", "../data/minist1_lcn_label.bin");
    MLP mlp; 

    MLPLayer *firstLayer = new TanhLayer(data.getFeatureNumber(), 500);
    Logistic *secondLayer = new Logistic(500, data.getLabelNumber());
    mlp.addLayer(firstLayer);
    mlp.addLayer(secondLayer);

    TrainModel mlpModel(mlp);
    mlpModel.train(&data, 0.01, 20, 1000);
}

void testMNIST(){
    MNISTDataset mnist;
    mnist.loadData();
    MLP mlp; 

    SigmoidLayer *firstLayer = new SigmoidLayer(mnist.getFeatureNumber(), 500);
    SigmoidLayer *secondLayer = new SigmoidLayer(500, 500);
    Logistic *thirdLayer = new Logistic(500, mnist.getLabelNumber());
    mlp.addLayer(firstLayer);
    mlp.addLayer(secondLayer);
    mlp.addLayer(thirdLayer);
    mlp.setModelFile("result/MLPModel.dat");

    TrainModel mlpModel(mlp);
    mlpModel.train(&mnist, 0.01, 20, 3000);
}

void testMNISTLoading(){
    MNISTDataset mnist;
    mnist.loadData();
    MLP mlp("result/MLPModel.dat");

    TrainModel mlpModel(mlp);
    printf("validate error : %.8lf%%\n", 100.0 * mlpModel.getValidError(&mnist, 20));
}

int main(){
    //testWFICA();
    testMNIST();
    //testMNISTLoading();

    return 0;
}
