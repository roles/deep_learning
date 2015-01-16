#include "Dataset.h"
#include "TrainModel.h"
#include "DeepAutoEncoder.h"
#include <random>

using namespace std;

void testMNISTTraining(){
    MNISTDataset data;
    data.loadData();

    int sizes[] = {data.getFeatureNumber(), 500, 500};

    DeepAutoEncoder dad(2, sizes);
    TrainModel model(dad);
    model.train(&data, 0.01, 20, 15);
}

int main(){
    testMNISTTraining();
}
