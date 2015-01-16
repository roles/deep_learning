#include <cmath>
#include <cstdlib>

#ifndef _UTILITY_H
#define _UTILITY_H

double* I();
void initializeWeightSigmoid(double *weight, int numIn, int numOut);
void initializeWeightTanh(double *weight, int numIn, int numOut);
void softmax(double *arr, int size);
void multiNormial(double *px, double *x, int size);
void binomial(double *px, double *x, int size);
int maxElem(double *arr, int size);
double expc(double x);
double sigmoidc(double x);
double softplusc(double x);
double squareNorm(double *arr, int n, int size);
double normalize(double *arr, int n, int size);
double corrupt(const double* x, double* nx, int n, double level);
void transMatrix(double* ori, double* trans, int nrow, int ncol);

inline int random_int(int low, int high){
    return rand() % (high - low + 1) + low;
}

inline double random_double(double low, double high){
    return ((double)rand() / RAND_MAX) * (high - low) + low;
}

inline double sigmoid(double x){
    return 1.0 / (1.0 + exp(-x));
}

inline double softplus(double a){
    return log(1.0 + exp(a));
}

inline double get_sigmoid_derivative(double y){
    return (y + 1e-10) * (1.0 - y + 1e-10);
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
