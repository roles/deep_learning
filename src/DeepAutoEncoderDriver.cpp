#include "Dataset.h"
#include "TrainModel.h"
#include "DeepAutoEncoder.h"
#include "RBM.h"

using namespace std;

void testMNISTTraining(){
    MNISTDataset data;
    data.loadData();

    if(true){
        int sizes[] = {data.getFeatureNumber(), 1000, 500, 250, 2};
        DeepAutoEncoder dad(4, sizes);
        dad.setModelFile("result/dad_without_pretrain.dat");
        TrainModel model(dad);
        model.train(&data, 0.01, 10, 100);
    }

    if(false){
        int sizes[] = {data.getFeatureNumber(), 500};
        DeepAutoEncoder dad(1, sizes);
        TrainModel model(dad);
        model.train(&data, 0.01, 10, 100);
    }
}

void testMNISTFineTune(){
    MNISTDataset data;
    data.loadData();

    if(true){
        MultiLayerRBM* multirbm = new MultiLayerRBM("result/MNISTMultiLayerRBM_pretrain2.dat");
        DeepAutoEncoder dad(*multirbm); 
        delete multirbm;
        dad.setModelFile("result/dad_with_pretrain.dat");
        TrainModel model(dad);
        model.train(&data, 0.01, 10, 100);
    }

    if(false){
        MultiLayerRBM* multirbm = new MultiLayerRBM("result/MNISTMultiLayerRBM_pretrain_small.dat");
        DeepAutoEncoder dad(*multirbm); 
        delete multirbm;
        TrainModel model(dad);
        model.train(&data, 0.01, 10, 100);
    }
}

void testDump(){
    MNISTDataset data;
    data.loadData();
    DeepAutoEncoder dad("result/dad_with_pretrain.dat");

    TransmissionDataset out(data, dad);
    out.dumpTrainingData("result/MNIST_DeepAd.bin");
}

void testTCGATraining(){
    TCGADataset data;
    data.loadData("../data/TCGA/TCGA-gene-top2000-all.txt", "../data/TCGA/TCGA-gene-top2000-valid.txt");

    int sizes[] = {data.getFeatureNumber(), 1000, 500, 200, 2};
    DeepAutoEncoder dad(4, sizes);
    //dad.setModelFile("result/dad_without_pretrain.dat");
    TrainModel model(dad);
    model.train(&data, 0.01, 10, 100);
}

void testTCGAFineTune(){
    TCGADataset data;
    data.loadData("../data/TCGA/TCGA-gene-top2000-all.txt", "../data/TCGA/TCGA-gene-top2000-valid.txt");

    if(false){
        MultiLayerRBM* multirbm = new MultiLayerRBM("result/TCGAMultiLayerRBM_2000gene_4layer_2000_1000_5000_200_2.dat");
        DeepAutoEncoder dad(*multirbm); 
        delete multirbm;
        dad.setModelFile("result/TCGA_dad_4layer_2000_1000_5000_200_2.dat");
        TrainModel model(dad);
        model.train(&data, 0.01, 10, 1000);
    }

    if(true){
        DeepAutoEncoder dad("result/TCGA_dad_4layer_2000_1000_5000_200_2_600epoch.dat");
        dad.setModelFile("result/TCGA_dad_4layer_2000_1000_5000_200_2.dat");
        TrainModel model(dad);
        model.train(&data, 0.01, 10, 1000);
    }
}

void testDumpTCGA(){
    TCGADataset data;
    data.loadData("../data/TCGA/TCGA-gene-top2000-all.txt", "../data/TCGA/TCGA-gene-top2000-valid.txt");
    DeepAutoEncoder dad("result/TCGA_dad_4layer_2000_1000_5000_200_2.dat");

    TransmissionDataset out(data, dad);
    out.dumpTrainingData("result/TCGA_DeepAd.bin");
}

void debug(){
    MNISTDataset data;
    data.loadData();

    MultiLayerRBM* multirbm = new MultiLayerRBM("result/MNISTMultiLayerRBM_pretrain_small.dat");

    RBM* rbm = (*multirbm)[0];
    rbm->beforeTraining(10);
    rbm->setPersistent(false);
    rbm->setGaussianHidden(true);
    TrainModel rbmmodel(*rbm);
    printf("%lf\n", rbmmodel.getValidError(&data, 10));

    DeepAutoEncoder dad(*multirbm); 
    delete multirbm;

    TrainModel model(dad);
    dad.beforeTraining(10);
    printf("%lf\n", model.getValidError(&data, 10));
}

void debug2(){
    MNISTDataset data;
    data.loadData();

    RBM rbm(784, 500);
    rbm.setPersistent(false);
    rbm.setGaussianHidden(true);
    TrainModel model(rbm);
    model.train(&data, 0.001, 10, 5);

    printf("%lf\n", model.getValidError(&data, 10));
}

void debug3(){
    MNISTDataset data;
    data.loadData();

    RBM rbm(784, 500);
    rbm.setPersistent(false);
    rbm.setGaussianHidden(true);
    TrainModel model(rbm);
    model.train(&data, 0.001, 10, 5);

    printf("%lf\n", model.getValidError(&data, 10));
}

void testMNISTDeepADMaximization(){

}

int main(){
    //testMNISTTraining();
    //testMNISTFineTune();
    //testDump();
    //debug();
    //debug2();
    //testTCGATraining();
    testTCGAFineTune();
    //testDumpTCGA();
}
