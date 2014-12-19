#include "TrainModel.h"
#include "MultiLayerRBM.h"
#include "Utility.h"

void testMNISTTraining(){
    MNISTDataset mnist;
    mnist.loadData();
    int rbmLayerSize[] = { mnist.getFeatureNumber(), 1000, 1000, 1000};

    MultiLayerRBM multirbm(3, rbmLayerSize);
    multirbm.setModelFile("result/MNISTMultiLayerRBM_1000_1000_1000_0.01.dat");
    multirbm.setPersistent(true);

    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&mnist, 0.01, 10, 100);

    MLP mlp;
    multirbm.toMLP(&mlp, mnist.getLabelNumber());
    mlp.setModelFile("result/MNISTDBN_1000_1000_1000_0.1.dat");

    TrainModel supervisedModel(mlp);
    supervisedModel.train(&mnist, 0.1, 10, 1000);
}

void test20NewsgroupTraining(){
    SVMDataset data;
    data.loadData("../data/20newsgroup_train.txt", "../data/20newsgroup_valid.txt");
    
    int rbmLayerSize[] = { data.getFeatureNumber(), 2000, 2000, 1000};
    MultiLayerRBM multirbm(3, rbmLayerSize);
    /*
    MultiLayerRBM multirbm("result/20NewsgroupMultiLayerRBM_2000_1000_500_0.01.dat");
    multirbm.resetLayer(1, 2000, 2000);
    multirbm.resetLayer(2, 2000, 1000);
    */

    multirbm.setModelFile("result/20NewsgroupMultiLayerRBM_2000_2000_1000_0.01.dat");
    MultiLayerTrainModel pretrainModel(multirbm);
    int numEpochs[] = {100, 100, 100};
    pretrainModel.train(&data, 0.01, 10, numEpochs);

    MLP mlp;
    multirbm.toMLP(&mlp, data.getLabelNumber());
    mlp.setModelFile("result/20NewsgroupDBN_2000_2000_1000_0.1.dat");

    TrainModel supervisedModel(mlp);
    supervisedModel.train(&data, 0.1, 10, 1000);
}

void testDump20Newsgroup(){
    SVMDataset data;
    data.loadData("../data/20newsgroup_train.txt", "../data/20newsgroup_valid.txt");

    MLP* mlp = new MLP("result/20NewsgroupDBN_2000_1000_500_0.1.dat");
    MultiLayerRBM multirbm(*mlp);
    delete mlp;
    TrainComponent& firstLayer = multirbm.getLayer(0);
    TrainComponent& secondLayer = multirbm.getLayer(1);
    TrainComponent& thirdLayer = multirbm.getLayer(2);

    // dump first layer weight
    //firstLayer.setModelFile("result/20NewsgroupDBN_FirstLayer.dat");
    //firstLayer.saveModel();

    // dump first layer hidden layer output
    TransmissionDataset h1out(data, firstLayer);
    h1out.dumpTrainingData("result/20NewsgroupDBN_FirstLayer_Hidden.bin");

    // dump second layer hidden layer output
    TransmissionDataset h2out(h1out, secondLayer);
    h2out.dumpTrainingData("result/20NewsgroupDBN_SecondLayer_Hidden.bin");

    TransmissionDataset h3out(h2out, thirdLayer);
    h3out.dumpTrainingData("result/20NewsgroupDBN_ThirdLayer_Hidden.bin");

    /*
    double avgNorm = squareNorm(data.getTrainingData(0), data.getFeatureNumber(), 9000);
    // dump first layer AM weight
    multirbm.activationMaxization(0, 2000, avgNorm, 500, "result/20NewsgroupDBN_FirstLayer_AM.bin", true);

    // dump second layer AM weight
    multirbm.activationMaxization(1, 1000, avgNorm, 500, "result/20NewsgroupDBN_SecondLayer_AM.bin", true);

    // dump third layer AM weight
    multirbm.activationMaxization(2, 500, avgNorm, 500, "result/20NewsgroupDBN_ThirdLayer_AM.bin", true);
    */

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
    multirbm.setModelFile("result/TCGAMultiLayerRBM_0.01_13epoch.dat");
    multirbm.setPersistent(false);

    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&data, 0.01, 1, 13);

    MLP mlp;
    multirbm.toMLP(&mlp, data.getLabelNumber());
    mlp.setModelFile("result/TCGADBN_0.01.dat");

    TrainModel supervisedModel(mlp);
    supervisedModel.train(&data, 0.01, 1, 1000);
}

void testTCGALoading(){
    TCGADataset data;
    data.loadData();

    //MultiLayerRBM multirbm("result/TCGAMultiLayerRBM_0.01_13epoch.dat");
    MultiLayerRBM multirbm("result/TCGAMultiLayerRBM_2000_0.01_13epoch_2000_0.01_3epoch.dat");
    multirbm.setPersistent(false);
    MLP mlp;
    multirbm.toMLP(&mlp, data.getLabelNumber());
    mlp.setModelFile("result/TCGADBN_0.01.dat");

    TrainModel dbn(mlp);
    dbn.train(&data, 0.01, 1, 1000, 30);
}

void testTCGAUpperLayerTraining(){
    TCGADataset data;
    data.loadData();

    MultiLayerRBM multirbm("result/TCGAMultiLayerRBM_2000_0.01_13epoch_2000_0.01_3epoch.dat");
    multirbm.setModelFile("result/TCGAMultiLayerRBM_2000_0.01_13epoch_2000_0.01_3epoch_2000_0.01.dat");
    multirbm.addLayer(2000);
    multirbm.setLayerToTrain(0, false);
    multirbm.setLayerToTrain(1, false);

    MultiLayerTrainModel dbn(multirbm);
    dbn.train(&data, 0.01, 1, 3);
}

void testMNISTAM(){
    MNISTDataset mnist;
    mnist.loadData();
    double avgNorm = squareNorm(mnist.getTrainingData(0), mnist.getFeatureNumber(), 100);

    MultiLayerRBM multirbm("result/MNISTMultiLayerRBM_1000_1000_1000_0.01.dat");
    multirbm.activationMaxization(0, 400, avgNorm, 1000);
}

void testMNISTDBNAM(){
    MNISTDataset mnist;
    mnist.loadData();
    double avgNorm = squareNorm(mnist.getTrainingData(0), mnist.getFeatureNumber(), 100);

    MLP* mlp = new MLP("result/MNISTDBN_1000_1000_1000_0.1.dat");
    MultiLayerRBM multirbm(*mlp);
    delete mlp;
    multirbm.activationMaxization(2, 20, avgNorm, 1000);
}

int main(){
    srand(4321);
    //testMNISTTraining();
    //testMNISTLoading();
    //testMNISTDBNSecondLayerTrain();
    //testTCGATraining();
    //testTCGAUpperLayerTraining();
    //testTCGALoading();

    //testMNISTAM();
    //testMNISTDBNAM();

    //test20NewsgroupTraining();
    testDump20Newsgroup();
    return 0;
}
