#include "Config.h"
#include "TrainModel.h"
#include "Utility.h"
#include <ctime>

TrainModel::TrainModel(TrainComponent& comp) : component(comp) {}

void TrainModel::train(Dataset *data, double learningRate, int batchSize, int numEpoch){

    component.beforeTraining(batchSize);

    int numBatch = (data->getTrainingNumber()-1) / batchSize + 1;
    component.setLearningRate(learningRate);

    bool stop = false;
    double bestErr = 100;   // 用于记录early stopping
    int patience = 10000;   // 至少要运行的minibatch数量

    for(int epoch = 0; epoch < numEpoch && !stop; epoch++){
        time_t startTime = time(NULL);
        double cost = 0.0;

        for(int k = 0; k < numBatch; k++){
#ifdef DEBUG
            if(numBatchPerLog != 0 && (k+1) % numBatchPerLog == 0){
                printf("epoch %d batch %d\n", epoch + 1, k + 1);
                fflush(stdout);
            }
#endif
            int theBatchSize;

            if(k == (numBatch-1)){
                theBatchSize = data->getTrainingNumber() - batchSize * k;
            }else{
                theBatchSize = batchSize;
            }

            component.setInput(data->getTrainingData(k * batchSize));
            if(component.getTrainType() == Supervise){
                component.setLabel(data->getTrainingLabel(k * batchSize));
            }
            
            component.trainBatch(theBatchSize);

            if(component.getTrainType() == Unsupervise){
                cost += component.getTrainingCost(theBatchSize, numBatch);
            }
        }


        if(component.getTrainType() == Supervise){  //supervised model
            int iterCount = numBatch * (epoch + 1);     //已经运行minibatch数量

            double err = getValidError(data, batchSize);
            time_t endTime = time(NULL);
            printf("epoch %d validate error: %.8lf%%\ttime: %ds \n", epoch+1, err * 100, (int)(endTime - startTime));
            fflush(stdout);

            // early stopping
            if(err < bestErr){
                if(err < bestErr * 0.995){  //如果validate error有较大的提升，则增加patience
                    if(patience < iterCount * 2){
                        patience = iterCount * 2;
                        printf("patience update to %d, needs %d epochs\n", patience, (patience-1) / numBatch + 1);
                    }
                }
                bestErr = err;
            }
            if(iterCount > patience){
                stop = true;
            }
        }else { // unsupervise model
            time_t endTime = time(NULL);
            printf("epoch %d cost: %.8lf\ttime: %.2lf min\n", epoch+1, cost, (double)(endTime - startTime) / 60);
            fflush(stdout);
        }

        component.operationPerEpoch();
    }
    component.afterTraining(batchSize);
}

double TrainModel::getValidError(Dataset* data, int batchSize){
    int err = 0;
    int numBatch = (data->getValidateNumber()-1) / batchSize + 1;
    int numOut = data->getLabelNumber();

    for(int k = 0; k < numBatch; k++){

        int theBatchSize;

        if(k == (numBatch-1)){
            theBatchSize = data->getValidateNumber() - batchSize * k;
        }else{
            theBatchSize = batchSize;
        }

        component.setInput(data->getValidateData(k * batchSize));
        component.setLabel(data->getValidateLabel(k * batchSize));
        component.runBatch(theBatchSize);
        for(int i = 0; i < theBatchSize; i++){
            double *out = component.getOutput();
            int l = maxElem(out+i*numOut, numOut);
            if(*(component.getLabel() + i*numOut+l) != 1.0){
                err++;
            }
        }
    }
    return ((double)err) / data->getValidateNumber();
}

void MultiLayerTrainModel::train(Dataset* data, 
        double learningRate, int batchSize, int numEpoch)
{
    int numLayer = component.getLayerNumber();
    Dataset* curData = data;
    for(int i = 0; i < numLayer; i++){
        TrainModel model(component.getLayer(i));
        if(i != 0){
            Dataset* transData = new TransmissionDataset(*curData, component.getLayer(i-1));
            if(curData != data)
                delete curData;
            curData = transData;
        }
        printf("Training layer %d ********\n", i+1);
        fflush(stdout);
        model.train(curData, learningRate, batchSize, numEpoch);
    }
    if(curData != data)
        delete curData;
    component.saveModel();
}


