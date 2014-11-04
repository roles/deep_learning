#include "Config.h"
#include "Utility.h"

double* I(){
    static double _I[maxUnit*maxBatchSize];
    static bool hasVisit = false;
    if(!hasVisit){
        for(int i = 0; i < maxUnit*maxBatchSize; i++)
            _I[i] = 1.0;
        hasVisit = true;
    }
    return _I;
}

void initializeWeight(double *weight, int numIn, int numOut){
    double low, high; 

    low = -4 * sqrt((double)6 / (numIn + numOut));
    high = 4 * sqrt((double)6 / (numIn + numOut));

    for(int i = 0; i < numIn*numOut; i++){
        weight[i] = random_double(low, high);
    }
}

void softmax(double *arr, int size){
    double sum = 0.0;
    for(int i = 0; i < size; i++){
        arr[i] = exp(arr[i]);
        sum += arr[i];
    }

    for(int i = 0; i < size; i++)
        arr[i] /= sum;
}

int maxElem(double *arr, int size){
    int maxe = 0;
    for(int i = 1; i < size; i++){
        if(arr[i] > arr[maxe])
            maxe = i;
    }
    return maxe;
}
