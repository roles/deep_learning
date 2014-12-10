#include "ClassRBM.h"
#include "MultiLayerRBM.h"

#ifndef _DEEPCLASSRBM_H
#define _DEEPCLASSRBM_H

class DeepClassRBM : public MultiLayerTrainComponent {
    public:
        DeepClassRBM(MultiLayerRBM*, ClassRBM*);
        ~DeepClassRBM();
        TrainComponent& getLayer(int);
        int getLayerNumber();
        bool getLayerToTrain(int);
        void saveModel(FILE* fd);
    private:
        MultiLayerRBM* multirbm;
        ClassRBM* classrbm;
        int numLayer;
};

#endif
