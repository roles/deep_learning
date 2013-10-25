#include<strings.h>
#include "dataset.h"

#ifndef LOG_SGD_H
#define LOG_SGD_H

#define OUT_LAYER_SIZE 20
#define IN_LAYER_SIZE 1000
#define eta 0.13

typedef struct log_reg {
    int n_in, n_out; 
    double ** W;
    double * b;
} log_reg;

void init_log_reg(log_reg *m, int n_in, int n_out);
void free_log_reg(log_reg *m);
void dump_param(rio_t *rp, log_reg *m);
void load_log_reg(rio_t *rp, log_reg *m);
void get_softmax_y_given_x(const log_reg *m, const double *x, double *y);
void get_grad(const log_reg *m, const double *x, const double *y, const double *t, double **grad_w, double *grad_b);
double get_log_reg_delta(const double *y, const double *t, double *d, const int size);

#endif
