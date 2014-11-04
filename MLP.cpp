#include "Logistic.h"
#include <vector>

using namespace std;

class MLP {
    public:
        MLP(vector<MLPLayer> &l);
        void train(Dataset *data, double learningRate = 0.01, int batchSize = 20, int numEpoch = 1000);
    private:
        vector<MLPLayer> &layers;
        int numLayer;
};

void MLP::train(Dataset *data, double learningRate, int batchSize, int numEpoch){
    
}

MLP::MLP(vector<MLPLayer> &l) : layers(l){
    numLayer = layers.size();
}

int main(){
}
