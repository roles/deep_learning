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
        void addLayer(int);
        void saveModel(FILE*);
        void setPersistent(bool);
        void toMLP(MLP*, int);
        void loadLayer(int, const char*);
        void setLayerToTrain(int i, bool b) { layersToTrain[i] = b; }
        bool getLayerToTrain(int i) { return layersToTrain[i]; }
    private:
        RBM* layers[maxLayer];
        int numLayer;
        vector<bool> layersToTrain;
        bool persistent;
};
