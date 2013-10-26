#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<stdint.h>
#include<arpa/inet.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include"rio.h"

#ifndef DATASET_H
#define DATASET_H

#define DEFAULT_MAXSIZE 1000

typedef struct dataset{
    uint32_t N;
    uint32_t nrow, ncol;
    double **input;
    uint8_t *output;
} dataset;

typedef struct dataset_blas{
    uint32_t N;
    uint32_t nrow, ncol;
    double *input;
    uint8_t *output;
} dataset_blas;

void init_dataset(dataset *d, uint32_t N, uint32_t nrow, uint32_t ncol);
void load_dataset_input(rio_t *rp, dataset *d);
void load_dataset_output(rio_t *rp, dataset *d);
void print_dataset(const dataset *d);
void free_dataset(dataset *d);
void read_uint32(rio_t *rp, uint32_t *data);
int random_int(int low, int high);
double random_double(double low, double high);
double sigmoid(double x);
double get_sigmoid_derivative(double y);
double tanh(double x);
double get_tanh_derivative(double y);
void load_mnist_dataset(dataset *train_set, dataset *validate_set);

void init_dataset_blas(dataset_blas *d, uint32_t N, uint32_t nrow, uint32_t ncol);
void load_dataset_blas_input(rio_t *rp, dataset_blas *d);
void load_dataset_blas_output(rio_t *rp, dataset_blas *d);
void free_dataset_blas(dataset_blas *d);
void print_dataset_blas(const dataset_blas *d);
void load_mnist_dataset_blas(dataset_blas *train_set, dataset_blas *validate_set);

#endif
