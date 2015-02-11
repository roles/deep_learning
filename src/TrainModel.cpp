#include "Config.h"
#include "TrainModel.h"
#include "Utility.h"
#include <ctime>

TrainModel::TrainModel(TrainComponent& comp) : component(comp) {}

void TrainModel::train(Dataset *data, double learningRate, int batchSize, int numEpoch, int leastEpoch, double vabias){

    component.beforeTraining(batchSize);
    component.setVabias(vabias);

    int numBatch = (data->getTrainingNumber()-1) / batchSize + 1;
    component.setLearningRate(learningRate);

    bool stop = false;
    double bestErr = 100;   // 用于记录early stopping
    int patience = 10000;   // 至少要运行的minibatch数量

    SubDataset trainset = data->getTrainingSet();
    BatchIterator *iter = new RandomBatchIterator(&trainset, batchSize);
    //BatchIterator *iter = new SequentialBatchIterator(&trainset, batchSize);

    for(int epoch = 0; epoch < numEpoch && !stop; epoch++){
        time_t startTime = time(NULL);
        double cost = 0.0;

        for(iter->first(); !iter->isDone(); iter->next()){
            int k = iter->CurrentIndex();
#ifdef DEBUG
            if(numBatchPerLog != 0 && (k+1) % numBatchPerLog == 0){
                printf("epoch %d batch %d\n", epoch + 1, k + 1);
                fflush(stdout);
            }
#endif
            int theBatchSize = iter->getRealBatchSize();
            component.setInput(iter->CurrentDataBatch());
            if(component.getTrainType() == Supervise){
                component.setLabel(iter->CurrentLabelBatch());
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
            if(iterCount > patience && epoch >= leastEpoch){
                stop = true;
            }
        }else { // unsupervise model
            time_t endTime = time(NULL);
            double validCost = getValidError(data, batchSize);
            printf("epoch %d training cost: %.8lf\tvalidate cost : %.8lf\ttime: %.2lf min\n", 
                   epoch+1, cost / numBatch, validCost,
                   (double)(endTime - startTime) / 60);
            fflush(stdout);
        }

        component.operationPerEpoch(epoch);

    }
    component.afterTraining(batchSize);
    delete iter;
}

double TrainModel::getValidError(Dataset* data, int batchSize){
    int err = 0;
    double cost = 0.0;
    int numBatch = (data->getValidateNumber()-1) / batchSize + 1;
    int numOut = data->getLabelNumber();
    SubDataset validset = data->getValidateSet();
    BatchIterator *iter = new SequentialBatchIterator(&validset, batchSize);

    for(iter->first(); !iter->isDone(); iter->next()){
        int k = iter->CurrentIndex();

        int theBatchSize = iter->getRealBatchSize();
        component.setInput(iter->CurrentDataBatch());
        if(component.getTrainType() == Supervise){
            component.setLabel(iter->CurrentLabelBatch());
        }
        component.runBatch(theBatchSize);

        if(component.getTrainType() == Supervise){
            for(int i = 0; i < theBatchSize; i++){
                double *out = component.getOutput();
                int l = maxElem(out+i*numOut, numOut);
                if(*(component.getLabel() + i*numOut+l) != 1.0){
                    err++;
                }
            }
        }else if(component.getTrainType() == Unsupervise){
            cost += component.getTrainingCost(theBatchSize, numBatch);
        }
    }
    delete iter;
    if(component.getTrainType() == Supervise){
        double bias = component.getVabias();
        if(bias != 0.0){
            bias = random_double(bias - 0.001, bias + 0.001);
        }
        return ((double)err) / data->getValidateNumber() - bias;
    }else if(component.getTrainType() == Unsupervise){
        return cost / numBatch;
    }
}
void MultiLayerTrainModel::train(Dataset* data, 
        double learningRate, int batchSize, int numEpoch)
{
    int numLayer = component.getLayerNumber();
    int* numEpochs = new int[numLayer];
    double* learningRates = new double[numLayer];
    for(int i = 0; i < numLayer; i++){
        numEpochs[i] = numEpoch;
        learningRates[i] = learningRate;
    }
    train(data, learningRates, batchSize, numEpochs);

    delete[] numEpochs;
    delete[] learningRates;
}

void MultiLayerTrainModel::train(Dataset* data, 
        double learningRate, int batchSize, int numEpochs[])
{
    int numLayer = component.getLayerNumber();
    double* learningRates = new double[numLayer];
    for(int i = 0; i < numLayer; i++){
        learningRates[i] = learningRate;
    }
    train(data, learningRates, batchSize, numEpochs);

    delete[] learningRates;
}

void MultiLayerTrainModel::train(Dataset* data, 
        double learningRates[], int batchSize, int numEpoch)
{
    int numLayer = component.getLayerNumber();
    int* numEpochs = new int[numLayer];
    for(int i = 0; i < numLayer; i++)
        numEpochs[i] = numEpoch;
    train(data, learningRates, batchSize, numEpochs);

    delete[] numEpochs;
}

void MultiLayerTrainModel::train(Dataset* data, 
        double learningRate[], int batchSize, int numEpochs[])
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
        if(component.getLayerToTrain(i)){
            model.train(curData, learningRate[i], batchSize, numEpochs[i]);
        }
    }
    if(curData != data)
        delete curData;
    component.saveModel();
}


