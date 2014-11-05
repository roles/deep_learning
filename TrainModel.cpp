#include "TrainModel.h"
#include "Utility.h"
#include <ctime>

TrainComponent::TrainComponent(TrainType t) : trainType(t) {}

TrainComponent::~TrainComponent(){}

TrainModel::TrainModel(TrainComponent& comp) : component(comp) {}

void TrainModel::train(Dataset *data, double learningRate, int batchSize, int numEpoch){

    int numBatch = (data->getTrainingNumber()-1) / batchSize + 1;
    component.setLearningRate(learningRate);

    for(int epoch = 0; epoch < numEpoch; epoch++){
        time_t startTime = time(NULL);

        for(int k = 0; k < numBatch; k++){
#ifdef DEBUG
            if((k+1) % 500 == 0){
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
        }

        if(component.getTrainType() == Supervise){
            double err = getValidError(data, batchSize);
            time_t endTime = time(NULL);
            printf("epoch %d error: %.8lf%%\ttime: %ds \n", epoch+1, err * 100, (int)(endTime - startTime));
        }
    }
}

double TrainModel::getValidError(Dataset* data, int batchSize){
    int err = 0;
    int numBatch = (data->getValidateNumber()-1) / batchSize + 1;
    double *out = component.getOutput();
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
            int l = maxElem(out+i*numOut, numOut);
            if(*(component.getLabel() + i*numOut+l) != 1.0){
                err++;
            }
        }
    }
    return ((double)err) / data->getValidateNumber();
}
