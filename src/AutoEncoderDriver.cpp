#include "AutoEncoder.h"
#include "Dataset.h"
#include "TrainModel.h"

int main(){
    MNISTDataset data;
    data.loadData();

    AutoEncoder ad(data.getFeatureNumber(), 500, false);
    TrainModel model(ad);
    model.train(&data, 0.1, 20, 15);
}
