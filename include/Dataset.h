extern "C" {
#include <stdint.h>
}
#include <cstdio>
#include <cstring>

#ifndef _DATASET_H_
#define _DATASET_H_

class Dataset {
    public:
        Dataset();
        
        inline double* getTrainingData(int sampleOffset){
            return trainingData + numFeature * sampleOffset;
        }
        inline double* getTrainingLabel(int sampleOffset){
            return trainingLabel + numLabel * sampleOffset;
        }
        inline double* getValidateData(int sampleOffset){
            return validateData + numFeature * sampleOffset;
        }
        inline double* getValidateLabel(int sampleOffset){
            return validateLabel + numLabel * sampleOffset;
        }
        inline int getTrainingNumber(){
            return numTrain;
        }
        inline int getValidateNumber(){
            return numValid;
        }
        inline int getFeatureNumber(){
            return numFeature;
        }
        inline int getLabelNumber(){
            return numLabel;
        }
        virtual ~Dataset();

    protected:
        int numFeature, numLabel;
        int numTrain, numValid, numTest;

        double  *trainingData;
        double  *validateData;
        double  *testData;
        double  *trainingLabel;
        double  *validateLabel;
        double  *testLabel;
};

class TrivialDataset : public Dataset {
    public:
        void loadData(const char *trainingDataFile, const char *trainingLabelFile);
        ~TrivialDataset();
};

class MNISTDataset : public Dataset {
    public:
        void loadData(const char *trainingDataFile = "../data/train-images-idx3-ubyte", 
                const char *trainingLabelFile = "../data/train-labels-idx1-ubyte");
        ~MNISTDataset();
};

#endif
