#include "Logistic.h"

int main(){
    MNISTDataset mnist;
    mnist.loadData();

    Logistic logi(mnist.getFeatureNumber(), mnist.getLabelNumber());
    TrainModel logisticModel(logi);
    logisticModel.train(&mnist, 0.13, 500, 1000);
}
