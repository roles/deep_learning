#include <cmath>
#include <cstdlib>

#ifndef _UTILITY_H
#define _UTILITY_H

double* I();
void initializeWeightSigmoid(double *weight, int numIn, int numOut);
void initializeWeightTanh(double *weight, int numIn, int numOut);
void softmax(double *arr, int size);
int maxElem(double *arr, int size);

inline int random_int(int low, int high){
    return rand() % (high - low + 1) + low;
}

inline double random_double(double low, double high){
    return ((double)rand() / RAND_MAX) * (high - low) + low;
}

inline double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

inline double get_sigmoid_derivative(double y){
    return y * (1.0 - y);
}

inline double square(double x){
    return x * x;
}

inline double tanh(double x){
    double a = square(exp(x));
    return (a - 1.0) / (a + 1.0);
}

inline double get_tanh_derivative(double y){
    return 1.0 - y * y;
}

#endif
