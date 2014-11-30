#include "TrainModel.h"
#include "MultiLayerRBM.h"

void testMNISTTraining(){
    MNISTDataset mnist;
    mnist.loadData();
    int rbmLayerSize[] = { mnist.getFeatureNumber(), 500, 500};

    MultiLayerRBM multirbm(2, rbmLayerSize);
    multirbm.setModelFile("result/MultiLayerRBM.dat");
    multirbm.setPersistent(false);

    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&mnist, 0.01, 10, 20);

    MLP mlp;
    multirbm.toMLP(&mlp, mnist.getLabelNumber());
    mlp.setModelFile("result/DBN.dat");

    TrainModel supervisedModel(mlp);
    supervisedModel.train(&mnist, 0.1, 10, 1000);
}

void testMNISTLoading(){
    MNISTDataset mnist;
    mnist.loadData();

    MultiLayerRBM multirbm("result/MultiLayerRBM.dat");
    MLP mlp;
    multirbm.toMLP(&mlp, mnist.getLabelNumber());
    mlp.setModelFile("result/DBN.dat");

    TrainModel dbn(mlp);
    dbn.train(&mnist, 0.1, 10, 1000);
}

void testMNISTDBNSecondLayerTrain(){
    MNISTDataset mnist;
    mnist.loadData();

    int rbmLayerSize[] = { mnist.getFeatureNumber(), 500, 500};

    MultiLayerRBM multirbm(2, rbmLayerSize);
    multirbm.setModelFile("result/MultiLayerRBM.dat");
    multirbm.loadLayer(0, "result/DBN_Layer1.dat");
    multirbm.setLayerToTrain(0, false);

    MultiLayerTrainModel dbn(multirbm);
    dbn.train(&mnist, 0.01, 20, 50);
}

void testTCGATraining(){
    TCGADataset data;
    data.loadData();
    int rbmLayerSize[] = { data.getFeatureNumber(), 2000};

    MultiLayerRBM multirbm(1, rbmLayerSize);
    multirbm.setModelFile("result/TCGAMultiLayerRBM_0.01_20epoch.dat");
    multirbm.setPersistent(false);

    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&data, 0.01, 1, 20);

    MLP mlp;
    multirbm.toMLP(&mlp, data.getLabelNumber());
    mlp.setModelFile("result/DBN_0.01.dat");

    TrainModel supervisedModel(mlp);
    supervisedModel.train(&data, 0.1, 1, 1000);
}

int main(){
    srand(1234);
    //testMNISTTraining();
    //testMNISTLoading();
    //testMNISTDBNSecondLayerTrain();
    testTCGATraining();
    return 0;
}
