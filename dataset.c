#include<string.h>
#include<strings.h>
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

void init_dataset_blas(dataset_blas *d, uint32_t N, uint32_t nrow, uint32_t ncol){
    int i;
    d->N = N;
    d->nrow = nrow;
    d->ncol = ncol;
    d->input = (double*)malloc(d->N * d->nrow * d->ncol * sizeof(double));
    d->output = (uint8_t*)malloc(d->N * sizeof(uint8_t));
}

void init_dataset_blas_simple(dataset_blas *d, int N, int n_feature){
    d->N = N;
    d->n_feature = n_feature;
    d->input = (double*)malloc(N * n_feature * sizeof(double));
    d->output = (uint8_t*)malloc(N * sizeof(uint8_t));
    d->label = NULL;
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

void load_dataset_blas_input(rio_t *rp, dataset_blas *d){
    int i, j, k;
    int idx;
    uint8_t pixel;

#ifdef DEBUG
    printf("loading input data...\n");
#endif
    for(i = 0; i < d->N; i++){
        for(j = 0; j < d->nrow * d->ncol; j++){
            idx = i * d->nrow * d->ncol + j;
            rio_readnb(rp, &pixel, sizeof(uint8_t));
            d->input[idx] = (double)pixel / 255.0;
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

void load_dataset_blas_output(rio_t *rp, dataset_blas *d){
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

void print_dataset_blas(const dataset_blas *d){
    int i, j, k; 
    int idx;
    int show_n = 5;
    for(i = 0; i < show_n; i++){
        printf("input %d\n", i + 1);
        for(j = 0; j < d->ncol * d->nrow; j++){
            idx = i * d->ncol * d->nrow + j;
            printf("%.2lf%s", d->input[idx], ((j+1) % d->ncol == 0) ? "\n" : " ");
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

void free_dataset_blas(dataset_blas *d){
    if(d->input != NULL){
        free(d->input);
    }
    if(d->output != NULL){
        free(d->output);
    }
    if(d->label != NULL){
        free(d->label);
    }
}

void read_uint32(rio_t *rp, uint32_t *data){
    rio_readnb(rp, data, sizeof(uint32_t));
    *data = ntohl(*data);
}

void load_mnist_dataset(dataset *train_set, dataset *validate_set){
    uint32_t N, nrow, ncol, magic_n;
    rio_t rio_train_x, rio_train_y;
    int train_x_fd, train_y_fd;
    int train_set_size = 50000, validate_set_size = 10000;

    train_x_fd = open("../data/train-images-idx3-ubyte", O_RDONLY);
    train_y_fd = open("../data/train-labels-idx1-ubyte", O_RDONLY);

    if(train_x_fd == -1){
        fprintf(stderr, "cannot open train-images-idx3-ubyte\n");
        exit(1);
    }
    if(train_y_fd == -1){
        fprintf(stderr, "cannot open train-labels-idx1-ubyte\n");
        exit(1);
    }

    rio_readinitb(&rio_train_x, train_x_fd, 0);
    rio_readinitb(&rio_train_y, train_y_fd, 0);

    read_uint32(&rio_train_x, &magic_n);
    read_uint32(&rio_train_x, &N);
    read_uint32(&rio_train_x, &nrow);
    read_uint32(&rio_train_x, &ncol);
    
    read_uint32(&rio_train_y, &magic_n);
    read_uint32(&rio_train_y, &N);
#ifdef DEBUG
    printf("magic number: %u\nN: %u\nnrow: %u\nncol: %u\n", magic_n, N, nrow, ncol);
    fflush(stdout);
#endif

    init_dataset(train_set, train_set_size, nrow, ncol);
    init_dataset(validate_set, validate_set_size, nrow, ncol);

    load_dataset_input(&rio_train_x, train_set);
    load_dataset_output(&rio_train_y, train_set);

    load_dataset_input(&rio_train_x, validate_set);
    load_dataset_output(&rio_train_y, validate_set);

    //print_dataset(&validate_set);

    close(train_x_fd);
    close(train_y_fd);
}


void load_mnist_dataset_blas(dataset_blas *train_set, dataset_blas *validate_set){
    uint32_t N, nrow, ncol, magic_n;
    rio_t rio_train_x, rio_train_y;
    int train_x_fd, train_y_fd;
    int train_set_size = 50000, validate_set_size = 10000;

    train_x_fd = open("../data/train-images-idx3-ubyte", O_RDONLY);
    train_y_fd = open("../data/train-labels-idx1-ubyte", O_RDONLY);

    if(train_x_fd == -1){
        fprintf(stderr, "cannot open train-images-idx3-ubyte\n");
        exit(1);
    }
    if(train_y_fd == -1){
        fprintf(stderr, "cannot open train-labels-idx1-ubyte\n");
        exit(1);
    }

    rio_readinitb(&rio_train_x, train_x_fd, 0);
    rio_readinitb(&rio_train_y, train_y_fd, 0);

    read_uint32(&rio_train_x, &magic_n);
    read_uint32(&rio_train_x, &N);
    read_uint32(&rio_train_x, &nrow);
    read_uint32(&rio_train_x, &ncol);
    
    read_uint32(&rio_train_y, &magic_n);
    read_uint32(&rio_train_y, &N);
#ifdef DEBUG
    printf("magic number: %u\nN: %u\nnrow: %u\nncol: %u\n", magic_n, N, nrow, ncol);
    fflush(stdout);
#endif

    init_dataset_blas(train_set, train_set_size, nrow, ncol);
    init_dataset_blas(validate_set, validate_set_size, nrow, ncol);

    load_dataset_blas_input(&rio_train_x, train_set);
    load_dataset_blas_output(&rio_train_y, train_set);

    load_dataset_blas_input(&rio_train_x, validate_set);
    load_dataset_blas_output(&rio_train_y, validate_set);

    //print_dataset(&validate_set);

    close(train_x_fd);
    close(train_y_fd);
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

void load_tcga_dataset_blas(dataset_blas *train_set, char *filename){
    FILE *f;
    int i, j;

    f = fopen(filename, "r");
    if(f == NULL){
        fprintf(stderr, "cannot open %s\n", filename);
        exit(1);
    }

    fscanf(f, "%d", &train_set->N);
    fscanf(f, "%d", &train_set->n_feature);    

    train_set->input = (double*)malloc(train_set->N * train_set->n_feature * sizeof(double));
    for(i = 0; i < train_set->N; i++){
        for(j = 0; j < train_set->n_feature; j++){
            fscanf(f, "%lf", &train_set->input[i * train_set->n_feature + j]);
        }
    }
    printf("data loaded\n");
    train_set->output = NULL;

    fclose(f);
}


void partition_trainset(dataset_blas *all_trainset, 
                        dataset_blas *foldset, int fold_num){
    int i, j, k;    
    int fold_size = all_trainset->N / fold_num;
    for(i = 0; i < fold_num; i++){
        if(i == fold_num - 1){
            foldset[i].N = all_trainset->N - fold_size * (fold_num - 1);
        }else{
            foldset[i].N = fold_size;
        }
        foldset[i].n_feature = all_trainset->n_feature;
        foldset[i].input = all_trainset->input + (i * fold_size * all_trainset->n_feature); 
    }
}

void combine_foldset(dataset_blas *foldset, int fold_num, int valid_fold,
                     dataset_blas *train_set, dataset_blas *validate_set){
    int i, j, k, p;
    int all_num = 0;

    for(i = 0; i < fold_num; i++){
        all_num += foldset[i].N;
    }
    train_set->N = all_num - foldset[valid_fold].N;
    train_set->n_feature = foldset[valid_fold].n_feature;

    for(i = 0, k = 0; i < fold_num; i++){
        if(i != valid_fold){
            memcpy(train_set->input + k, foldset[i].input, 
                   foldset[i].N * train_set->n_feature * sizeof(double));
            k += foldset[i].N * train_set->n_feature;
        }
    }

    validate_set = &foldset[valid_fold];
}

void load_corpus(char* filename, dataset_blas* train_set){
    FILE* f;
    int N, K;
    char *str;
    int len, nread;
    //buffer need to be large
    char line[100000];
    char *token;
    int i = 0;
    int idx, cnt;

    if((f = fopen(filename, "r")) == NULL){
        fprintf(stderr, "cannot open %s\n", filename);
        exit(1);
    }
    fscanf(f, "%d%d", &N, &K);
    fgetc(f);
    init_dataset_blas_simple(train_set, N, K);
    bzero(train_set->input, N * K * sizeof(double));
    
    while(fgets(line, sizeof(line), f)){
        token = strtok(line, " ");
        while(token != NULL){
            sscanf(token, "%d", &idx);
            str = strchr(token, ':');
            sscanf(str+1, "%d", &cnt);
            //printf("%d %d\n", idx, cnt);
            train_set->input[i * train_set->n_feature + idx - 1] = cnt;
            token = strtok(NULL, " ");
        }
        i++;
    }

    fclose(f);
}

void load_corpus_label(char *filename, dataset_blas *train_set){
    FILE *f;
    int i, j, k;

    if((f = fopen(filename, "r")) == NULL){
        fprintf(stderr, "cannot open %s\n", filename);
        exit(1);
    }

    fscanf(f, "%d", &train_set->nlabel);
    train_set->label = (double*)malloc(train_set->N * train_set->nlabel * sizeof(double));
    for(i = 0; i < train_set->N; i++){
        fscanf(f, "%d", &k);
        for(j = 0; j < train_set->nlabel; j++){
            if((j+1) == k){
                train_set->label[i*train_set->nlabel+j] = 1;
            }else{
                train_set->label[i*train_set->nlabel+j] = 0;
            }
        }
    }
}

void shuffle(int *arr, int n){
    int i, j;
    int t;

    for(i = n-1; i > 0; i--){
        j = rand() % i;
        t = arr[j];
        arr[j] = arr[i];
        arr[i] = t;
    }
}
