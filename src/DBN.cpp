#include "TrainModel.h"
#include "RBM.h"
#include "MLP.h"
#include <vector>

using namespace std;

class MultiLayerRBM : public MultiLayerTrainComponent {
    public:
        MultiLayerRBM(int, const int[]);
        MultiLayerRBM(int, const vector<const char*> &);   //从多个单层模型文件读取
        MultiLayerRBM(const char*);
        ~MultiLayerRBM();
        TrainComponent& getLayer(int);
        int getLayerNumber() { return numLayer; }
        void saveModel(FILE*);
        void toMLP(MLP*, int);
    private:
        RBM* layers[maxLayer];
        int numLayer;
};

MultiLayerRBM::MultiLayerRBM(int numLayer, const int layersSize[]) :
    MultiLayerTrainComponent("MultiLayerRBM"), numLayer(numLayer)
{
    char weightFile[100], modelFile[20];
    for(int i = 0; i < numLayer; i++){
        layers[i] = new RBM(layersSize[i], layersSize[i+1]); 
        layers[i]->setPersistent(false);

        //sprintf(weightFile, "result/DBN_Layer%d_weight.txt", i+1);
        //sprintf(modelFile, "result/DBN_Layer%d.dat", i+1);
        //layers[i]->setWeightFile(weightFile);
        //layers[i]->setModelFile(modelFile);
    }
}

MultiLayerRBM::MultiLayerRBM(int numLayer, const vector<const char*> &layerModelFiles) :
    MultiLayerTrainComponent("MultiLayerRBM"), numLayer(numLayer)
{
    for(int i = 0; i < numLayer; i++){
        layers[i] = new RBM(layerModelFiles[i]);
        layers[i]->setPersistent(false);
    }
}

MultiLayerRBM::MultiLayerRBM(const char* file) :
    MultiLayerTrainComponent("MultiLayerRBM")
{
    FILE* fd = fopen(file, "rb");
    if(fd == NULL){
        fprintf(stderr, "file not exist : %s\n", file);
        exit(1);
    }
    fread(&numLayer, sizeof(int), 1, fd);
    for(int i = 0; i < numLayer; i++){
        layers[i] = new RBM(fd);
        layers[i]->setPersistent(false);
    }
    fclose(fd);
}

void MultiLayerRBM::saveModel(FILE* fd){
    fwrite(&numLayer, sizeof(int), 1, fd);
    for(int i = 0; i < numLayer; i++){
        layers[i]->saveModel(fd);
    }
}

MultiLayerRBM::~MultiLayerRBM(){
    for(int i = 0; i < numLayer; i++){
        delete layers[i];
    }
}

TrainComponent& MultiLayerRBM::getLayer(int i){
    return *layers[i];
}

void MultiLayerRBM::toMLP(MLP* mlp, int lastNumOut){
    mlp->setLayerNumber(numLayer);
    for(int i = 0; i < numLayer; i++){
        mlp->setLayer(i, new SigmoidLayer(layers[i]->numVis, layers[i]->numHid, layers[i]->weight, layers[i]->hbias));
    }
    mlp->addLayer(new Logistic(layers[numLayer-1]->numHid, lastNumOut));
}

void testMNISTTraining(){
    MNISTDataset mnist;
    mnist.loadData();
    int rbmLayerSize[] = { mnist.getFeatureNumber(), 500, 500};

    MultiLayerRBM multirbm(2, rbmLayerSize);
    MultiLayerTrainModel dbn(multirbm);
    dbn.train(&mnist, 0.01, 10, 20);
}

void testMNISTLoading(){
    MNISTDataset mnist;
    mnist.loadData();

    vector<const char*> layerModelFiles;
    layerModelFiles.push_back("result/DBN_Layer1.dat");
    layerModelFiles.push_back("result/DBN_Layer2.dat");

    MultiLayerRBM multirbm(2, layerModelFiles);
    MLP mlp;
    multirbm.toMLP(&mlp, mnist.getLabelNumber());
    mlp.setModelFile("result/DBN.dat");

    TrainModel dbn(mlp);
    dbn.train(&mnist, 0.1, 10, 1000);
}

int main(){
    srand(1235);
    testMNISTTraining();
    //testMNISTLoading();
    return 0;
}
