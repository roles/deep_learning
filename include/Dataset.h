extern "C" {
#include <stdint.h>
}
#include <cstdio>
#include <cstring>
#include <vector>

using namespace std;

#ifndef _DATASET_H_
#define _DATASET_H_

class TrainComponent;
class BatchIterator;


class SubDataset {
    public:
        SubDataset(int numSample, int numFeature, int numLabel,
                   double *data, double *label) 
            : numSample(numSample), numFeature(numFeature),
              numLabel(numLabel), data(data), label(label) { }
    private:
        int numSample, numFeature, numLabel;
        double *data;
        double *label;
        friend class SequentialBatchIterator;
        friend class RandomBatchIterator;
};

class BatchIterator {
    public:
        BatchIterator(SubDataset* data, int batchSize) : 
            data(data), batchSize(batchSize) { }
        virtual void first() = 0;
        virtual void next() = 0;
        virtual bool isDone() = 0;
        virtual int CurrentIndex() = 0;
        virtual double* CurrentDataBatch() = 0;
        virtual double* CurrentLabelBatch() = 0;
        virtual int getRealBatchSize() = 0;
    protected:
        SubDataset* data;
        int batchSize;
};


class SequentialBatchIterator : public BatchIterator {
    public:
        SequentialBatchIterator(SubDataset* data, int batchSize) :
            BatchIterator(data, batchSize)
        {
            size = (data->numSample-1) / batchSize + 1;
        }
        void first() { cur = 0; }
        bool isDone() { return cur >= size; }
        void next() { cur++; }
        int CurrentIndex() { return cur; }

        double* CurrentDataBatch() { 
            return data->data + data->numFeature * cur * batchSize;
        }
        double* CurrentLabelBatch() { 
            return data->label + data->numLabel* cur * batchSize;
        }
        int getRealBatchSize();

    private:
        int cur;
        int size;
};

class RandomBatchIterator : public BatchIterator {
    public:
        RandomBatchIterator(SubDataset *data, int batchSize);
        void first();
        bool isDone() { return cur >= size; }
        void next() { cur++; }
        int CurrentIndex() { return cur; }

        double* CurrentDataBatch() { 
            return data->data + data->numFeature * randIndex[cur] * batchSize;
        }
        double* CurrentLabelBatch() { 
            return data->label + data->numLabel* randIndex[cur] * batchSize;
        }
        int getRealBatchSize();

    private:
        int size, cur;
        vector<int> randIndex;
};

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
        SubDataset getTrainingSet();
        SubDataset getValidateSet();
        SubDataset getTestSet();
        void dumpTrainingData(const char[]);
        void rowNormalize();

    protected:
        void dumpData(const char[], int numData, double* data, double *label);
        int numFeature, numLabel;
        int numTrain, numValid, numTest;

        double  *trainingData;
        double  *validateData;
        double  *testData;
        double  *trainingLabel;
        double  *validateLabel;
        double  *testLabel;
};

class TransmissionDataset : public Dataset {
    public:
        TransmissionDataset(Dataset& originData, TrainComponent& component);
        ~TransmissionDataset();
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

class SVMDataset : public Dataset {
    public:
        virtual void loadData(const char [], const char []);
        ~SVMDataset(){ }
};

class TCGADataset : public SVMDataset {
    public:
        void loadData(const char tcgaTrainDataFile[] = "../data/TCGA/TCGA-gene-top5000-train.txt", 
                      const char tcgaValidDataFile[] = "../data/TCGA/TCGA-gene-top5000-valid.txt");
        ~TCGADataset(){ }
};


#endif
