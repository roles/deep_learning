#include "TrainModel.h"
#include "MultiLayerRBM.h"
#include "Utility.h"

void testMNISTTraining(){
    MNISTDataset mnist;
    mnist.loadData();
    int rbmLayerSize[] = { mnist.getFeatureNumber(), 1000, 1000, 1000};

    MultiLayerRBM multirbm(3, rbmLayerSize);
    multirbm.setModelFile("result/MNISTMultiLayerRBM_1000_1000_1000_0.01_sparsity.dat");
    multirbm.setPersistent(true);
    multirbm.setSparsity(true, 0.001, 0.99, 0.1);

    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&mnist, 0.01, 10, 100);

    MLP mlp;
    multirbm.toMLP(&mlp, mnist.getLabelNumber());
    mlp.setModelFile("result/MNISTDBN_1000_1000_1000_0.1_sparsity.dat");

    TrainModel supervisedModel(mlp);
    supervisedModel.train(&mnist, 0.1, 10, 1000);
}

void testMNISTGuassian(){
    MNISTDataset mnist;
    mnist.loadData();
    mnist.rowNormalize();

    int rbmLayerSize[] = { mnist.getFeatureNumber(), 500};
    MultiLayerRBM multirbm(1, rbmLayerSize);
    multirbm.setModelFile("result/MNISTMultiLayerRBM_gaussian.dat");
    multirbm.setPersistent(false);
    //multirbm.setGaussian(true);

    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&mnist, 0.01, 10, 20);

    MLP mlp;
    multirbm.toMLP(&mlp, mnist.getLabelNumber());
    mlp.setModelFile("result/MNISTDBN_gaussian.dat");
    //mlp.setGaussian(true);

    TrainModel supervisedModel(mlp);
    supervisedModel.train(&mnist, 0.01, 10, 1000);
}

void testMNISTPretrain(){
    MNISTDataset mnist;
    mnist.loadData();

    if(true){
        double lr[] = { 0.01, 0.01, 0.01, 0.005 };
        int rbmLayerSize[] = { mnist.getFeatureNumber(), 1000, 500, 250, 2};
        MultiLayerRBM multirbm(4, rbmLayerSize);
        /*
        int rbmLayerSize[] = { mnist.getFeatureNumber(), 500};
        MultiLayerRBM multirbm(1, rbmLayerSize);
        */
        multirbm.setModelFile("result/MNISTMultiLayerRBM_pretrain2.dat");
        multirbm.setPersistent(false);
        multirbm.setGaussianHidden(3, true);

        MultiLayerTrainModel pretrainModel(multirbm);
        pretrainModel.train(&mnist, lr, 10, 100);
    }

    if(false){
        double lr[] = { 0.001 };
        int rbmLayerSize[] = { mnist.getFeatureNumber(), 500};
        MultiLayerRBM multirbm(1, rbmLayerSize);

        multirbm.setModelFile("result/MNISTMultiLayerRBM_pretrain_small.dat");
        multirbm.setPersistent(false);
        multirbm.setGaussianHidden(0, true);

        MultiLayerTrainModel pretrainModel(multirbm);
        pretrainModel.train(&mnist, lr, 10, 50);
    }
}

void testGPCRGuassian(){
    TrivialDataset data;
    data.loadData("../data/GPCR/GPCR_Binary.data", "../data/GPCR/GPCR_Binary.label");
    data.rowNormalize();

    /*
    int rbmLayerSize[] = { data.getFeatureNumber(), 500};
    MultiLayerRBM multirbm(1, rbmLayerSize);
    multirbm.setModelFile("result/GPCRMultiLayerRBM_gaussian.dat");
    multirbm.setPersistent(false);
    multirbm.setGaussian(true);

    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&data, 0.01, 5, 20);
    */

    MultiLayerRBM multirbm("result/GPCRMultiLayerRBM_gaussian.dat");
    multirbm.addLayer(500);
    multirbm.setLayerToTrain(0, false);
    multirbm.setModelFile("result/GPCRMultiLayerRBM_SecondLayer_gaussian.dat");
    multirbm.setPersistent(false);
    multirbm.setGaussianVisible(0, true);

    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&data, 0.01, 5, 20);

    MLP mlp;
    multirbm.toMLP(&mlp, data.getLabelNumber());
    mlp.setModelFile("result/GPCRDBN_gaussian.dat");
    //mlp.setGaussian(true);

    TrainModel supervisedModel(mlp);
    supervisedModel.train(&data, 0.01, 1, 1000, 40);
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

void testDumpTCGA(){
    TCGADataset data;
    data.loadData();

    MLP* mlp = new MLP("result/TCGADBN_ThreeLayer_2000_0.01.dat");
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
    h1out.dumpTrainingData("result/TCGADBN_FirstLayer_Hidden.bin");

    // dump second layer hidden layer output
    TransmissionDataset h2out(h1out, secondLayer);
    h2out.dumpTrainingData("result/TCGADBN_SecondLayer_Hidden.bin");

    TransmissionDataset h3out(h2out, thirdLayer);
    h3out.dumpTrainingData("result/TCGADBN_ThirdLayer_Hidden.bin");

    double avgNorm = squareNorm(data.getTrainingData(0), data.getFeatureNumber(), 2000);
    // dump first layer AM weight
    multirbm.activationMaxization(0, 2000, avgNorm, 500, "result/TCGADBN_FirstLayer_AM.bin", true);

    // dump second layer AM weight
    multirbm.activationMaxization(1, 2000, avgNorm, 500, "result/TCGADBN_SecondLayer_AM.bin", true);

    // dump third layer AM weight
    multirbm.activationMaxization(2, 2000, avgNorm, 500, "result/TCGADBN_ThirdLayer_AM.bin", true);
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
    int rbmLayerSize[] = { data.getFeatureNumber(), 3000};

    MultiLayerRBM multirbm("result/TCGAMultiLayerRBM_FirstLayer_2000_0.01.dat");
    //MultiLayerRBM multirbm(1, rbmLayerSize);
    multirbm.setPersistent(false);
    multirbm.setModelFile("result/TCGAMultiLayerRBM_FirstLayer_2000.dat");
    multirbm.setLayerToTrain(0, false);

    /*
    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&data, 0.01, 1, 12);
    */

    MLP mlp;
    multirbm.toMLP(&mlp, data.getLabelNumber());
    mlp.setModelFile("result/TCGADBN_OneLayer_3000_0.01.dat");

    TrainModel supervisedModel(mlp);
    supervisedModel.train(&data, 0.005, 1, 1000, 40, 0.001);
}


void testTCGALoading(){
    srand(1111);
    TCGADataset data;
    data.loadData();

    MultiLayerRBM multirbm("result/TCGAMultiLayerRBM_SecondLayer_2000_0.01.dat");
    //MultiLayerRBM multirbm("result/TCGAMultiLayerRBM_ThirdLayer_2000_0.01.dat");

    MLP mlp;
    multirbm.toMLP(&mlp, data.getLabelNumber());
    mlp.setModelFile("result/TCGADBN_TwoLayer_2000_0.01.dat");

    TrainModel dbn(mlp);
    dbn.train(&data, 0.01, 1, 41, 41);
}

void testTCGAUpperLayerTraining(){
    TCGADataset data;
    data.loadData();

    MultiLayerRBM multirbm("result/TCGAMultiLayerRBM_SecondLayer_2000_0.01.dat");
    multirbm.setModelFile("result/TCGAMultiLayerRBM_ThirdLayer_2000_0.01.dat");
    multirbm.addLayer(2000);
    multirbm.setLayerToTrain(0, false);
    multirbm.setLayerToTrain(1, false);

    MultiLayerTrainModel dbn(multirbm);
    dbn.train(&data, 0.01, 1, 12);
}

void testMNISTAM(){
    MNISTDataset mnist;
    mnist.loadData();
    double avgNorm = squareNorm(mnist.getTrainingData(0), mnist.getFeatureNumber(), 100);

    MultiLayerRBM multirbm("result/MNISTMultiLayerRBM_1000_1000_1000_0.01.dat");

    multirbm.activationMaxization(0, 400, avgNorm, 1000, "result/MNISTDBN_FirstLayer_AM.txt");

    multirbm.activationMaxization(1, 400, avgNorm, 1000, "result/MNISTDBN_SecondLayer_AM.txt");

    multirbm.activationMaxization(2, 400, avgNorm, 1000, "result/MNISTDBN_ThirdLayer_AM.txt");
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

void testTCGAPretrain(){
    TCGADataset data;
    data.loadData("../data/TCGA/TCGA-gene-top2000-all.txt", "../data/TCGA/TCGA-gene-top2000-valid.txt");
    double lr[] = { 0.01, 0.01, 0.01, 0.005 };
    int rbmLayerSize[] = { data.getFeatureNumber(), 1000, 500, 200, 2};

    MultiLayerRBM multirbm(4, rbmLayerSize);
    multirbm.setModelFile("result/TCGAMultiLayerRBM_2000gene_4layer_2000_1000_5000_200_2.dat");
    multirbm.setPersistent(false);
    multirbm.setGaussianHidden(3, true);

    MultiLayerTrainModel pretrainModel(multirbm);
    pretrainModel.train(&data, lr, 1, 100);
}

void testPancanTraing(){
    SVMDataset data;
    data.loadData("../data/TCGA/Pancan-GAM-train.txt", "../data/TCGA/Pancan-GAM-valid.txt");

    //one layer pretrain
    if(true){
        int rbmLayerSize[] = { data.getFeatureNumber(), 200};
        MultiLayerRBM multirbm(1, rbmLayerSize);
        multirbm.setPersistent(false);
        multirbm.setModelFile("result/DBN-Pancan-GAM-onelayer-pretrain.dat");

        MultiLayerTrainModel pretrainModel(multirbm);
        pretrainModel.train(&data, 0.01, 5, 100);

        MLP mlp;
        multirbm.toMLP(&mlp, data.getLabelNumber());
        mlp.setModelFile("result/DBN-Pancan-GAM-onelayer.dat");

        TrainModel supervisedModel(mlp);
        supervisedModel.train(&data, 0.1, 5, 1000, 40);
    }

    //one layer pretrain
    if(true){
        MultiLayerRBM multirbm("result/DBN-Pancan-GAM-onelayer-pretrain.dat");

        MLP mlp;
        multirbm.toMLP(&mlp, data.getLabelNumber());
        mlp.setModelFile("result/DBN-Pancan-GAM-onelayer.dat");

        TrainModel supervisedModel(mlp);
        supervisedModel.train(&data, 0.01, 5, 1000, 40);
    }
}

int main(){
    srand(4321);
    //testMNISTTraining();
    //testMNISTLoading();
    //testMNISTDBNSecondLayerTrain();
    //testTCGATraining();
    //testTCGAUpperLayerTraining();
    //testTCGALoading();
    //testDumpTCGA();

    //testMNISTAM();
    //testMNISTDBNAM();

    //test20NewsgroupTraining();
    //testDump20Newsgroup();

    //testMNISTGuassian();
    //testGPCRGuassian();
    //testMNISTPretrain();
    
    //testTCGAPretrain();

    
    return 0;
}
