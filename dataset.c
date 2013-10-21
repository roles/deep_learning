#include "dataset.h"

void init_dataset(dataset *d, uint32_t N, uint32_t nrow, uint32_t ncol){
    int i;
    d->N = N;
    d->nrow = nrow;
    d->ncol = ncol;
    d->input = (double**)malloc(d->N * sizeof(double*));
    for(i = 0; i < d->N; i++){
        d->input[i] = (double*)malloc(d->nrow * d->ncol * sizeof(double));
    }
    d->output = (uint8_t*)malloc(d->N * sizeof(uint8_t));
}

void load_dataset_input(rio_t *rp, dataset *d){
    int i, j, k;
    int idx;
    uint8_t pixel;

#ifdef DEBUG
    printf("loading input data...\n");
#endif
    for(i = 0; i < d->N; i++){
        for(j = 0; j < d->nrow; j++){
            for(k = 0; k < d->ncol; k++){
                idx = j * d->nrow + k;     
                rio_readnb(rp, &pixel, sizeof(uint8_t));
                d->input[i][idx] = (double)pixel / 255.0;
            }
        }
    }
#ifdef DEBUG
    printf("data loaded\n");
    fflush(stdout);
#endif
}

void load_dataset_output(rio_t *rp, dataset *d){
    int i;
    uint8_t label;
    
    for(i = 0; i < d->N; i++)
        rio_readnb(rp, &d->output[i], sizeof(uint8_t));
}

void print_dataset(const dataset *d){
    int i, j, k; 
    int idx;
    int show_n = 5;
    for(i = 0; i < show_n; i++){
        printf("input %d\n", i + 1);
        for(j = 0; j < d->nrow; j++){
            for(k = 0; k < d->ncol; k++){
                idx = j * d->nrow + k;
                printf("%.2lf%s", d->input[i][idx], (k == d->ncol - 1) ? "\n" : " ");
            }
        }
        printf("output %d: %u\n", i + 1, d->output[i]);
    }
}

void free_dataset(dataset *d){
    int i;
    for(i = 0; i < d->N; i++){
        free((uint8_t*)d->input[i]);
    }
    free((uint8_t**)d->input);
}

void read_uint32(rio_t *rp, uint32_t *data){
    rio_readnb(rp, data, sizeof(uint32_t));
    *data = ntohl(*data);
}

int random_int(int low, int high){
    return rand() % (high - low + 1) + low;
}

double random_double(double low, double high){
    return ((double)rand() / RAND_MAX) * (high - low) + low;
}

double sigmoid(double x){
    double ret;
    ret = 1.0 / (1.0 + exp(-x));
    return ret;
}

double get_sigmoid_derivative(double y){
    return y * (1.0 - y);
}

double tanh(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double get_tanh_derivative(double y){
    return 1.0 - y * y;
}
