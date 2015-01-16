#include "Dataset.h"
#include "TrainComponent.h"
#include "MultiLayerTrainComponent.h"

#ifndef _TRAINMODEL_H
#define _TRAINMODEL_H


class TrainModel {
    public:
        TrainModel(TrainComponent&);
        void train(Dataset *, double, int, int, int leastEpoch = 1, double vabias = 0.0);
        double getValidError(Dataset *, int);
    private:
        TrainComponent& component;
};

class MultiLayerTrainModel {
    public:
        MultiLayerTrainModel(MultiLayerTrainComponent& comp) :
            component(comp){ }
        void train(Dataset *, double, int, int);
        void train(Dataset *, double, int, int[]);

    private:
        void trainOneLayer();
        MultiLayerTrainComponent& component;
};

/**
 * @brief  先预训练再进行有监督训练的训练模型
 */
class PretrainSupervisedModel {
    public:
        void pretrain(Dataset *, double, int, int);
        void supervisedTrain(Dataset *, double, int, int);
};

#endif
