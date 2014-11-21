#include "ClassRBM.h"
#include "Dataset.h"
#include "TrainModel.h"

int main(){
    MNISTDataset mnist;
    mnist.loadData();

    ClassRBM classrbm(mnist.getFeatureNumber(), 500, mnist.getLabelNumber()); //500个隐藏结点

    TrainModel ClassRBMModel(classrbm);
    ClassRBMModel.train(&mnist, 0.05, 20, 1000);

    return 0;
}
