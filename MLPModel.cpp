#include "Dataset.h"
#include "MLP.h"

void testWFICA(){
    TrivialDataset data;
    data.loadData("", "");
    MLP mlp; 

    MLPLayer *firstLayer = new MLPLayer(data.getFeatureNumber(), 500, Tanh);
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

    MLPLayer *firstLayer = new MLPLayer(mnist.getFeatureNumber(), 500, Tanh);
    Logistic *secondLayer = new Logistic(500, mnist.getLabelNumber());
    mlp.addLayer(firstLayer);
    mlp.addLayer(secondLayer);

    TrainModel mlpModel(mlp);
    mlpModel.train(&mnist, 0.01, 20, 1000);
}

int main(){
    testWFICA();

    return 0;
}
