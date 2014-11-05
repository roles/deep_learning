#include "Dataset.h"
#include "MLP.h"

int main(){
    MNISTDataset mnist;
    mnist.loadData();
    MLP mlp; 

    MLPLayer *firstLayer = new MLPLayer(mnist.getFeatureNumber(), 500, Tanh);
    Logistic *secondLayer = new Logistic(500, mnist.getLabelNumber());
    mlp.addLayer(firstLayer);
    mlp.addLayer(secondLayer);

    TrainModel mlpModel(mlp);
    mlpModel.train(&mnist, 0.01, 20, 1000);

    return 0;
}
