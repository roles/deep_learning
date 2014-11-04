#include "Dataset.h"
#include "TrainModel.h"
#include "MLPLayer.h"
#include "Logistic.h"
#include <vector>

using namespace std;

class MLP : public TrainComponent{
    public:
        MLP();
        ~MLP();
        void trainBatch(int);
        void runBatch(int);
        void setLearningRate(double lr);
        void setInput(double *input);
        void setLabel(double *label);
        double* getOutput();
        double* getLabel();

        inline void addLayer(MLPLayer* l) { layers[numLayer++] = l; }
    private:
        MLPLayer* layers[maxLayer];
        int numLayer;

        double learningRate;
        double *label;
};

MLP::MLP() : TrainComponent(Supervise), numLayer(0) {}

MLP::~MLP(){
    for(int i = 0; i < numLayer; i++){
        delete layers[i];
    }
}

void MLP::trainBatch(int size){
    for(int i = 0; i < numLayer; i++){
        if(i != 0){
            layers[i]->setInput(layers[i-1]->getOutput());
        }
        layers[i]->forward(size);
    }

    for(int i = numLayer-1; i >= 0; i--){
        if(i == numLayer-1){
            layers[i]->backpropagate(size, NULL);
        }else{
            layers[i]->backpropagate(size, layers[i+1]);
        }
    }
}

void MLP::runBatch(int size){

    for(int i = 0; i < numLayer; i++){
        if(i != 0){
            layers[i]->setInput(layers[i-1]->getOutput());
        }
        layers[i]->forward(size);
    }
}

void MLP::setLearningRate(double lr){
    for(int i = 0; i < numLayer; i++){
        layers[i]->setLearningRate(lr);
    }
}

void MLP::setInput(double *input){
    layers[0]->setInput(input);
}

void MLP::setLabel(double *label){
    layers[numLayer-1]->setLabel(label);
}

double* MLP::getOutput(){
    return layers[numLayer-1]->getOutput();
}

double* MLP::getLabel(){
    return layers[numLayer-1]->getLabel();
}

MLP mlp; 

int main(){
    MNISTDataset mnist;
    mnist.loadData();

    MLPLayer *firstLayer = new MLPLayer(mnist.getFeatureNumber(), 500);
    Logistic *secondLayer = new Logistic(500, mnist.getLabelNumber());
    mlp.addLayer(firstLayer);
    mlp.addLayer(secondLayer);

    TrainModel mlpModel(mlp);
    //mlpModel.train(&mnist, 0.01, 20, 1000);
    mlpModel.train(&mnist, 0.01, 20, 10);

    return 0;
}
