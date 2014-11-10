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
    for(int i = 0; i < numLayer; i++){
        layers[i] = new RBM(layersSize[i], layersSize[i+1]); 
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
