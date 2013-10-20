#include "my_logistic_sgd.h"
#include<time.h>

#define eta 0.13

void init_log_reg(log_reg *m, int n_in, int n_out){
    int i;

    m->n_in = n_in;
    m->n_out = n_out;
    m->W = (double**)malloc(n_in * sizeof(double*));
    for(i = 0; i < n_out; i++){
        m->W[i] = (double*)calloc(n_in, sizeof(double));
    }
    m->b = (double*)calloc(n_out, sizeof(double));
}

void free_log_reg(log_reg *m){
    int i;
    
    for(i = 0; i < m->n_out; i++){
        free(m->W[i]);
    }
    free(m->W);
    free(m->b);
}

/**
 *  文件的结构
 *  uint32  n_in
 *  uint32  n_out      
 *  以下是n_out行，n_in列的参数矩阵W
 *  double  W[0][0]
 *  double  W[0][1]
 *  ...
 *  ...
 *  double  W[n_out-1][n_in-1]
 *  以下是偏置向量b
 *  double  b[0]
 *  ...
 *  ...
 *  double  b[n_out-1]
 */
void dump_param(rio_t *rp, log_reg *m){
    int i, j;    

    rio_writenb(rp, &m->n_in, sizeof(uint32_t));
    rio_writenb(rp, &m->n_out, sizeof(uint32_t));

    for(i = 0; i < m->n_out; i++){
        for(j = 0; j < m->n_in; j++)
            rio_writenb(rp, &m->W[i][j], sizeof(double));
    }
    for(i = 0; i < m->n_out; i++){
        rio_writenb(rp, &m->b[i], sizeof(double));
    }
}

void load_log_reg(rio_t *rp, log_reg *m){
    int i, j;     
    uint32_t n_in, n_out;

    rio_readnb(rp, &n_in, sizeof(uint32_t));
    rio_readnb(rp, &n_out, sizeof(uint32_t));

    init_log_reg(m, n_in, n_out);
    for(i = 0; i < m->n_out; i++){
        for(j = 0; j < m->n_in; j++)
            rio_readnb(rp, &m->W[i][j], sizeof(double));
    }
    for(i = 0; i < m->n_out; i++){
        rio_readnb(rp, &m->b[i], sizeof(double));
    }
}

void get_loss_error(const log_reg *m, const double **x, const uint8_t *t, const int size, double *loss, int *error){
    int i, j, k; 
    int pred_y;
    double y[OUT_LAYER_SIZE];
    double target[OUT_LAYER_SIZE], s;
    double max_y;
    
    *loss = 0.0;
    *error = 0;
    bzero(target, sizeof(target));
    for(i = 0; i < size; i++){
        get_softmax_y_given_x(m, x[i], y);
        for(j = 0, max_y = -1.0; j < m->n_out; j++){
            if(y[j] > max_y){
                max_y = y[j];
                pred_y = j;
            }
        }
        target[t[i]] = 1.0;
        for(j = 0, s = 0.0; j < m->n_out; j++){
            s += target[j] * log(y[j]); 
        }
        target[t[i]] = 0.0;
        *loss += s;
        *error += (pred_y == t[i] ? 0 : 1);
    }
    *loss = -(*loss) / size;
}

void get_softmax_y_given_x(const log_reg *m, const double *x, double *y){
    int i, j, k;
    double a[OUT_LAYER_SIZE], s; 

    for(j = 0, s = 0.0; j < m->n_out; j++){
        a[j] = 0.0;
        for(k = 0; k < m->n_in; k++){
            a[j] += m->W[j][k] * x[k];
        }
        s += exp(a[j]);
    }
    for(j = 0; j < m->n_out; j++){
        y[j] = exp(a[j]) / s; 
    }
}

void get_grad(const log_reg *m, const double *x, const double *y, const double *t, double **grad_w, double *grad_b){
    int i, j;
    double *delta;

    delta = (double*)malloc(m->n_out * sizeof(double));
    get_log_reg_delta(y, t, delta, m->n_out);
    for(i = 0; i < m->n_out; i++){
        for(j = 0; j < m->n_in; j++){
            grad_w[i][j] = delta[i] * x[j];
        }
        grad_b[i] = delta[i];
    }
    free(delta);
}

double get_log_reg_delta(const double *y, const double *t, double *d, const int size){
    int i;

    for(i = 0; i < size; i++){
        d[i] = y[i] - t[i];
    }
}

void train_log_reg(){
    int i, j, k, p, q;
    int train_x_fd, train_y_fd;
    int train_set_size = 50000, validate_set_size = 10000;
    int mini_batch = 500;
    int epcho, n_epcho = 10;
    
    double grad_b[OUT_LAYER_SIZE];
    double **grad_w;
    double delta_w[OUT_LAYER_SIZE][IN_LAYER_SIZE], delta_b[OUT_LAYER_SIZE];
    double prob_y[OUT_LAYER_SIZE];
    double target[OUT_LAYER_SIZE];

    uint32_t N, nrow, ncol, magic_n;
    dataset train_set, validate_set;
    rio_t rio_train_x, rio_train_y;
    log_reg m;
    time_t start_time, end_time;
    
    double loss;
    int error;

    int param_fd;
    rio_t rio_param;

    train_x_fd = open("../data/train-images-idx3-ubyte", O_RDONLY);
    train_y_fd = open("../data/train-labels-idx1-ubyte", O_RDONLY);
    param_fd = open("log_sgd.param", O_TRUNC | O_WRONLY);

    if(train_x_fd == -1){
        fprintf(stderr, "cannot open train-images-idx3-ubyte\n");
        exit(1);
    }
    if(train_y_fd == -1){
        fprintf(stderr, "cannot open train-labels-idx1-ubyte\n");
        exit(1);
    }
    if(param_fd == -1){
        fprintf(stderr, "cannot open log_sgd.param\n");
        exit(1);
    }

    rio_readinitb(&rio_train_x, train_x_fd, 0);
    rio_readinitb(&rio_train_y, train_y_fd, 0);

    rio_readinitb(&rio_param, param_fd, 1);

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

    init_dataset(&train_set, train_set_size, nrow, ncol);
    init_dataset(&validate_set, validate_set_size, nrow, ncol);

    load_dataset_input(&rio_train_x, &train_set);
    load_dataset_output(&rio_train_y, &train_set);

    load_dataset_input(&rio_train_x, &validate_set);
    load_dataset_output(&rio_train_y, &validate_set);

    //print_dataset(&validate_set);
    
    init_log_reg(&m, 784, 10);

    bzero(target, sizeof(target));
#ifdef DEBUG
    //get_loss_error(&m, validate_set.input, validate_set.output, validate_set_size, &loss, &error);
    //printf("origin loss :%.5lf\t error :%d\n", loss, error);
#endif
    grad_w = (double**)malloc(m.n_out * sizeof(double*));
    for(i = 0; i < m.n_out; i++)
        grad_w[i] = (double*)malloc(m.n_in * sizeof(double));


    start_time = time(NULL);
    for(epcho = 0; epcho < n_epcho; epcho++){

        for(k = 0; k < train_set.N / mini_batch; k++){
            bzero(delta_w, sizeof(delta_w));
            bzero(delta_b, sizeof(delta_b));
            for(i = 0; i < mini_batch; i++){

                /* 求gradient */
                get_softmax_y_given_x(&m, train_set.input[k*mini_batch+i], prob_y);
                target[train_set.output[k*mini_batch+i]] = 1.0;
                get_grad(&m, train_set.input[k*mini_batch+i], prob_y, target, grad_w, grad_b);
                target[train_set.output[k*mini_batch+i]] = 0.0;
                
                for(j = 0; j < m.n_out; j++){                   /* mini-batch累加delta */
                    for(p = 0; p < m.n_in; p++){
                        delta_w[j][p] += eta * grad_w[j][p];
                    }
                    delta_b[j] += eta * grad_b[j];
                }
            }

            for(j = 0; j < m.n_out; j++){                       /* 更新参数 */
                for(p = 0; p < m.n_in; p++){
                    m.W[j][p] -= delta_w[j][p] / mini_batch;
                }
                m.b[j] -= delta_b[j] / mini_batch;
            }
        }
#ifdef DEBUG
        get_loss_error(&m, validate_set.input, validate_set.output, validate_set_size, &loss, &error);
        printf("epcho %d loss :%.5lf\t error :%d\n", epcho + 1, loss, error);
#endif
    }
    end_time = time(NULL);
    printf("time: %d\n", (int)(end_time - start_time));

    //dump_param(&rio_param, &m);

    close(param_fd);
    close(train_x_fd);
    close(train_y_fd);
    
    for(i = 0; i < m.n_out; i++)
        free(grad_w[i]);
    free(grad_w);
    free_dataset(&train_set);
    free_log_reg(&m);
}

void test_load_param(){
    rio_t rio_param;
    int param_fd;
    log_reg m;

    param_fd = open("log_sgd.param", O_RDONLY);
    rio_readinitb(&rio_param, param_fd, 0);

    load_log_reg(&rio_param, &m);
    printf("n_in:%d\tn_out:%d\n", m.n_in, m.n_out);

    close(param_fd);
    free_log_reg(&m);
}
#if 1
int main(){
    train_log_reg();
    //test_load_param();
    return 0;
}
#endif
