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
}

void load_dataset_input(int fd, dataset *d){
    int i, j, k;
    int idx;
    uint8_t pixel;

#ifdef DEBUG
    printf("loading data...\n");
#endif
    for(i = 0; i < d->N; i++){
        for(j = 0; j < d->nrow; j++){
            for(k = 0; k < d->ncol; k++){
                idx = j * d->nrow + k;     
                read(fd, &pixel, sizeof(uint8_t));
                d->input[i][idx] = (double)pixel / 255.0;
            }
        }
    }
#ifdef DEBUG
    printf("data loaded\n");
    fflush(stdout);
#endif
}

void print_dataset(dataset *d){
    int i, j, k; 
    int idx;
    for(i = 0; i < 10; i++){
        for(j = 0; j < d->nrow; j++){
            for(k = 0; k < d->ncol; k++){
                idx = j * d->nrow + k;
                printf("%.2lf%s", d->input[i][idx], (k == d->ncol - 1) ? "\n" : " ");
            }
        }
    }
}

void free_dataset(dataset *d){
    int i;
    for(i = 0; i < d->N; i++){
        free((uint8_t*)d->input[i]);
    }
    free((uint8_t**)d->input);
}

void read_uint32(int fd, uint32_t *data){
    read(fd, data, sizeof(uint32_t));
    *data = ntohl(*data);
}
