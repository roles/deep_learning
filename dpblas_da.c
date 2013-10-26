#include"dataset.h"
#include"cblas.h"
#include<strings.h>
#include<time.h>

#define MAX_SIZE 1000
#define MAX_BATCH_SIZE 500
#define MAX_STEP 5000
#define eta 0.1

typedef struct da{
    int n_in, n_out;
    double *W;
    double *b;
    double *c;
} da;

double h_out[MAX_BATCH_SIZE * MAX_SIZE], z_out[MAX_BATCH_SIZE * MAX_SIZE]; 
double d_low[MAX_BATCH_SIZE * MAX_SIZE], d_high[MAX_BATCH_SIZE * MAX_SIZE];
double delta_W[MAX_SIZE*MAX_SIZE], delta_b[MAX_SIZE], delta_c[MAX_SIZE];
double Ivec[MAX_BATCH_SIZE * MAX_SIZE];
double tr1[MAX_BATCH_SIZE * MAX_SIZE], tr2[MAX_BATCH_SIZE * MAX_SIZE];
double cost[MAX_STEP];

void init_da(da *m, int n_in, int n_out){
    int i;
    double r;

    r = 4 * sqrt(6.0 / (n_in + n_out));

    m->n_in = n_in;
    m->n_out = n_out;
    m->W = (double*)malloc(m->n_out * m->n_in * sizeof(double));
    for(i = 0; i < m->n_out * m->n_in; i++){
        m->W[i] = random_double(-r, r);
    }
    m->b = (double*)calloc(m->n_out, sizeof(double));
    m->c = (double*)calloc(m->n_in, sizeof(double));
}

void free_da(da *m){
    free(m->W);
    free(m->b);
    free(m->c);
}

/**
 * @brief  
 *
 * @param m
 * @param x [batch_size * n_in] matrix
 * @param y [batch_size * n_out] matrix
 * @param batch_size
 */
void get_hidden_values(const da *m, const double *x, 
                       double *y, const int batch_size){
    int i;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->n_out, m->n_in,
                1, x, m->n_in, m->W, m->n_in,
                0, y, m->n_out);
    
    cblas_dger(CblasColMajor, batch_size, m->n_out,
               1.0, m->b, 1, Ivec, 1, y, m->n_out);

    for(i = 0; i < batch_size*m->n_out; i++){
        y[i] = sigmoid(y[i]);
    }
}

void get_reconstruct_input(const da *m, const double *x,
                           double *y, const int batch_size){
    int i;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, m->n_in, m->n_out,
                1, x, m->n_out, m->W, m->n_in,
                0, y, m->n_in);

    cblas_dger(CblasColMajor, batch_size, m->n_in,
               1.0, m->c, 1, Ivec, 1, y, m->n_in);

    /* TODO */
    for(i = 0; i < batch_size*m->n_in; i++){
        y[i] = sigmoid(y[i]);
    }
}

void get_top_delta(const da *m, const double *y, 
                   const double *x, double *d, const int batch_size){
    cblas_dcopy(batch_size * m->n_in, y, 1, d, 1);
    cblas_daxpy(batch_size * m->n_in, -1,
                x, 1, d, 1);
}

void get_second_delta(const da *m, const double *y_low, const double *d_high,
                      double *d_low, const int batch_size){
    int i;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->n_out, m->n_in,
                1, d_high, m->n_in, m->W, m->n_in,
                0, d_low, m->n_out);

    /* compute sigmoid derivative */
    cblas_dcopy(batch_size * m->n_out, Ivec, 1, tr1, 1);
    cblas_daxpy(batch_size * m->n_out, -1, y_low, 1, tr1, 1);
    cblas_dsbmv(CblasColMajor, CblasLower, batch_size * m->n_out,
                0, 1.0, tr1, 1, y_low, 1, 0.0, tr2, 1);

    cblas_dsbmv(CblasColMajor, CblasLower, batch_size * m->n_out,
                0, 1.0, tr2, 1, d_low, 1, 0.0, tr1, 1);
    cblas_dcopy(batch_size * m->n_out, tr1, 1, d_low, 1);
}

void train_da(da *m, dataset_blas *train_set, dataset_blas *expected_set, 
              int mini_batch, int n_epcho, char* weight_filename){
    int i, j, k, p, q;
    int epcho;
    double total_cost;
    time_t start_time, end_time;
    FILE *weight_file;

    //weight_file = fopen(weight_filename, "w");

    for(epcho = 0; epcho < n_epcho; epcho++){

        total_cost = 0.0;
        start_time = time(NULL);
        for(k = 0; k < train_set->N / mini_batch; k++){
            get_hidden_values(m, train_set->input + k * mini_batch, h_out, mini_batch);
            get_reconstruct_input(m, h_out, z_out, mini_batch);
            
            get_top_delta(m, z_out, expected_set->input + k * mini_batch, d_high, mini_batch);
            get_second_delta(m, h_out, d_high, d_low, mini_batch);

            /* modify weight matrix W */
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->n_out, m->n_in, mini_batch,
                        1, d_low, m->n_out,
                        train_set->input + k * mini_batch, m->n_in,
                        0, tr1, m->n_in);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->n_out, m->n_in, mini_batch,
                        1, h_out, m->n_out,
                        d_high, m->n_in, 0, tr2, m->n_in);

            cblas_daxpy(m->n_out * m->n_in, 1, tr1, 1, tr2, 1);
            
            cblas_daxpy(m->n_out * m->n_in, -eta / mini_batch, tr2, 1, m->W, 1);

            /* modify bias vector */
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->n_out, 1, mini_batch,
                        1.0 / mini_batch, d_low, m->n_out,
                        Ivec, 1, 0, tr1, 1);

            cblas_daxpy(m->n_out, -eta, tr1, 1, m->b, 1);
            
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->n_in, 1, mini_batch,
                        1.0 / mini_batch, d_high, m->n_in,
                        Ivec, 1, 0, tr1, 1);

            cblas_daxpy(m->n_in, -eta, tr1, 1, m->c, 1);
        }
    }

    //fclose(weight_file);
}

void test_da(){
    int i, j, k, p, q;
    int mini_batch = 20;
    int epcho, n_epcho = 15;
    int train_set_size = 50000, validate_set_size = 10000;
    dataset_blas train_set, validate_set; 
    da m;

    srand(1234);
    load_mnist_dataset_blas(&train_set, &validate_set);
    init_da(&m, 28*28, 500);

    for(i = 0; i < MAX_BATCH_SIZE * MAX_SIZE; i++)
        Ivec[i] = 1.0;

    train_da(&m, &train_set, &train_set, mini_batch, n_epcho, "da_blas_weight_origin.txt");

    //print_dataset_blas(&train_set);

    free_dataset_blas(&train_set);
    free_dataset_blas(&validate_set);
    free_da(&m);
}

int main(){
    test_da();
    return 0;
}
