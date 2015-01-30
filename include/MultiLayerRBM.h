#include "RBM.h"
#include "MLP.h"
#include <vector>

using namespace std;

#ifndef _MULTIRBM_H
#define _MULTIRBM_H

class DeepAutoEncoder;

class MultiLayerRBM : public MultiLayerTrainComponent {
    public:
        MultiLayerRBM(int, const int[]);
        MultiLayerRBM(int, const vector<const char*> &);   //从多个单层模型文件读取
        MultiLayerRBM(const char*);
        MultiLayerRBM(MLP&);
        MultiLayerRBM(DeepAutoEncoder&);
        ~MultiLayerRBM();
        TrainComponent& getLayer(int);
        int getLayerNumber() { return numLayer; }
        void addLayer(int);
        void resetLayer(int, int, int);
        void saveModel(FILE*);
        void setPersistent(bool);
        void toMLP(MLP*, int);
        void loadLayer(int, const char*);
        void setLayerToTrain(int i, bool b) { layersToTrain[i] = b; }
        bool getLayerToTrain(int i) { return layersToTrain[i]; }
        void activationMaxization(int layerIdx, int unitNum, double avgNorm, int nepoch = 1000, 
                const char amSampleFile[] = "result/AM.txt", bool dumpBinary = false);
        void activationMaxizationOneUnit(int layerIdx, int unitIdx, double avgNorm, int nepoch = 1000);
        void setSparsity(bool, double p = 0.0, double numda = 0.9, double slr = 0.01);
        RBM* operator[](int i){
            return layers[i];
        }
        void setGaussianVisible(int i, bool b) {
            layers[i]->setGaussianVisible(b);
        }
        void setGaussianHidden(int i, bool b) {
            layers[i]->setGaussianHidden(b);
        }
    private:
        double maximizeUnit(int layersIdx, int unitIdx, double*, double avgNorm, int nepoch);
        RBM* layers[maxLayer];
        int numLayer;
        vector<bool> layersToTrain;
        bool persistent;
        double *AMSample;
};

#endif
