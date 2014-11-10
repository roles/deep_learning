#include "TrainModel.h"
#include "RBM.h"
#include <vector>

using namespace std;

class MultiLayerRBM : public MultiLayerTrainComponent {
    public:
        MultiLayerRBM(int, const int[]);
        ~MultiLayerRBM();
        TrainComponent& getLayer(int);
        int getLayerNumber() { return numLayer; }
        void saveModel(FILE*);
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
        sprintf(weightFile, "result/DBN_Layer%d_weight.txt", i+1);
        sprintf(modelFile, "result/DBN_Layer%d.dat", i+1);
        layers[i]->setWeightFile(weightFile);
        layers[i]->setModelFile(modelFile);
    }
}

void MultiLayerRBM::saveModel(FILE* fd){

}

MultiLayerRBM::~MultiLayerRBM(){
    for(int i = 0; i < numLayer; i++){
        delete layers[i];
    }
}

TrainComponent& MultiLayerRBM::getLayer(int i){
    return *layers[i];
}

int main(){
    MNISTDataset mnist;
    mnist.loadData();
    int rbmLayerSize[] = { mnist.getFeatureNumber(), 500, 500};

    MultiLayerRBM multirbm(2, rbmLayerSize);
    MultiLayerTrainModel dbn(multirbm);
    dbn.train(&mnist, 0.01, 10, 20);
}
