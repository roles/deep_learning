#include "Config.h"
#include "Utility.h"

static double _I[maxUnit*maxBatchSize];

const double expThreshold = 30.0;

double* I(){
    static bool hasVisit = false;
    if(!hasVisit){
        for(int i = 0; i < maxUnit*maxBatchSize; i++)
            _I[i] = 1.0;
        hasVisit = true;
    }
    return _I;
}

void initializeWeightSigmoid(double *weight, int numIn, int numOut){
    double low, high; 

    low = -4 * sqrt((double)6 / (numIn + numOut));
    high = 4 * sqrt((double)6 / (numIn + numOut));

    for(int i = 0; i < numIn*numOut; i++){
        weight[i] = random_double(low, high);
    }
}

void initializeWeightTanh(double *weight, int numIn, int numOut){
    double low, high; 

    low = -sqrt((double)6 / (numIn + numOut));
    high = sqrt((double)6 / (numIn + numOut));

    for(int i = 0; i < numIn*numOut; i++){
        weight[i] = random_double(low, high);
    }
}

void softmax(double *arr, int size){
    double sum = 0.0;
    double maxval = -DBL_MAX;

    for(int i = 0; i < size; i++){
        if(maxval < arr[i])
            maxval = arr[i];
    }

    for(int i = 0; i < size; i++){
        arr[i] = exp(arr[i]-maxval);
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


/*
 * exp function with check to avoid `inf` and `nan`
 */
double expc(double x){
    if(x > expThreshold){
        return exp(expThreshold);
    }else if(x < -expThreshold){
        return exp(-expThreshold);
    }
    return exp(x);
}

/*
 * sigmoid function with check to avoid `inf` and `nan`
 */
double sigmoidc(double x){
    if(x > expThreshold){
        return 1.0;
    }else if(x < -expThreshold){
        return 0.0000000001;
    }
    return sigmoid(x);
}

/*
 * softplus function with check to avoid `inf` and `nan`
 */
double softplusc(double x){
    if(x > expThreshold){
        return softplus(expThreshold);
    }else if(x < -expThreshold){
        return softplus(-expThreshold);
    }
    return softplus(x);
}


