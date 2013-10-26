#include"dataset.h"
#include<strings.h>
#include<time.h>

#define MAX_SIZE 1000
#define eta 0.1

typedef struct da{
    int n_in, n_out;
    double **W;
    double *b;
    double *c;
} da;

double h_out[MAX_SIZE], z_out[MAX_SIZE]; 
double d_low[MAX_SIZE], d_high[MAX_SIZE];
double delta_W[MAX_SIZE][MAX_SIZE], delta_b[MAX_SIZE], delta_c[MAX_SIZE];

void init_da(da *m, int n_in, int n_out){
    int i, j; 
    double r;

    r = 4 * sqrt(6.0 / (n_in + n_out));

    m->n_in = n_in;
    m->n_out = n_out;
    m->W = (double**)malloc(m->n_out * sizeof(double*));
    for(i = 0; i < m->n_out; i++){
        m->W[i] = (double*)malloc(m->n_in * sizeof(double));
    }
    for(i = 0; i < m->n_out; i++){
        for(j = 0; j < m->n_in; j++){
            m->W[i][j] = random_double(-r, r);
        }
    }
    m->b = (double*)calloc(m->n_out, sizeof(double));
    m->c = (double*)calloc(m->n_in, sizeof(double));
}

void free_da(da *m){
    int i;

    for(i = 0; i < m->n_out; i++){
        free(m->W[i]);
    }
    free(m->W);
    free(m->b);
    free(m->c);
}

void get_hidden_values(const da *m, const double *x, double *y){
    int i, j; 
    double s, a;

    for(i = 0; i < m->n_out; i++){
        for(j = 0, a = 0.0; j < m->n_in; j++){
            a += m->W[i][j] * x[j]; 
        }
        a += m->b[i];
        y[i] = sigmoid(a);
    }
}

void get_reconstruct_input(const da *m, const double *x, double *y){
    int i, j;
    double a;

    for(i = 0; i < m->n_in; i++){
        for(j = 0, a = 0.0; j < m->n_out; j++){
            a += m->W[j][i] * x[j]; 
        }
        a += m->c[i];
        y[i] = sigmoid(a);
    }
}

void get_top_delta(const da *m, const double *y, const double *x, double *d){
    int i; 

    for(i = 0; i < m->n_in; i++){
        d[i] = y[i] - x[i];
    }
}

void get_second_delta(const da *m, const double *y_low, const double *d_high, double *d_low){
    int i, j, k;
    double s, derivative;

    for(i = 0; i < m->n_out; i++){
        derivative = get_sigmoid_derivative(y_low[i]);
        for(j = 0, s = 0.0; j < m->n_in; j++){
            s += d_high[j] * m->W[i][j];
        }
        d_low[i] = derivative * s;
    }
}

void corrupt_train_set_input(const dataset *train_set, 
                             double corrupt_level, dataset *corrupted_train_set){
    int i, j;
    double a = 0.0;
    int size = train_set->nrow * train_set->ncol;

    for(i = 0; i < train_set->N; i++){
        for(j = 0; j < size; j++){
            a = random_double(0.0, 1.0) < corrupt_level ? 0.0 : 1.0;
            corrupted_train_set->input[i][j] = a * train_set->input[i][j];
        }
    }
}

void dump_weight(FILE *f, const da *m, const int ncol){
    int i, j, k;
    for(i = 0; i < 100; i++){
        fprintf(f, "Hidden Node %d\n", i);
        for(j = 0; j < m->n_in; j++){
            fprintf(f, "%.5lf%s", m->W[i][j], (j+1) % ncol == 0 ? "\n" : "\t");
        }
        fflush(f);
    }
}

void train_da(da *m, dataset *train_set, dataset *expected_set, 
              int mini_batch, int n_epcho, char* weight_filename){
    int i, j, k, p, q;
    int epcho;
    double cost, total_cost;
    time_t start_time, end_time;
    FILE *weight_file;

    weight_file = fopen(weight_filename, "w");

    for(epcho = 0; epcho < n_epcho; epcho++){
        
        total_cost = 0.0;
        start_time = time(NULL);
        for(k = 0; k < train_set->N / mini_batch; k++){

            //if((k+1) % 500 == 0){
            //    printf("epcho %d batch %d\n", epcho + 1, k + 1);
            //}

            bzero(delta_W, sizeof(delta_W));
            bzero(delta_b, sizeof(delta_b));
            bzero(delta_c, sizeof(delta_c));
            cost = 0;

            for(i = 0; i < mini_batch; i++){

                /* feed-forward */
                get_hidden_values(m, train_set->input[k*mini_batch+i], h_out);
                get_reconstruct_input(m, h_out, z_out);

                /* back-propagation*/
                get_top_delta(m, z_out, expected_set->input[k*mini_batch+i], d_high);
                get_second_delta(m, h_out, d_high, d_low);

                for(j = 0; j < m->n_out; j++){
                    for(p = 0; p < m->n_in; p++){
                        delta_W[j][p] += d_low[j] * train_set->input[k*mini_batch+i][p] + d_high[p] * h_out[j];
                    }
                    delta_b[j] += d_low[j];
                }
                for(j = 0; j < m->n_in; j++){
                    delta_c[j] += d_high[j];
                }

                for(j = 0; j < m->n_in; j++){
                    cost -= expected_set->input[k*mini_batch+i][j] * log(z_out[j]) + (1.0 - expected_set->input[k*mini_batch+i][j]) * log(1.0 - z_out[j]);
                }
            }

            cost /= mini_batch;

            /* modify parameter */
            for(j = 0; j < m->n_out; j++){
                for(p = 0; p < m->n_in; p++){
                    m->W[j][p] -= eta * delta_W[j][p] / mini_batch;
                }
                m->b[j] -= eta * delta_b[j] / mini_batch;
            }
            for(j = 0; j < m->n_in; j++){
                m->c[j] -= eta * delta_c[j] / mini_batch;
            }
            
            total_cost += cost;
        }

        end_time = time(NULL);
        printf("epcho %d cost: %.5lf\ttime: %ds\n", epcho + 1, total_cost / train_set->N * mini_batch, (int)(end_time - start_time));
    }

    dump_weight(weight_file, m, 28);
    fclose(weight_file);
}

void test_da(){
    int i, j, k, p, q;
    int mini_batch = 20;
    int epcho, n_epcho = 15;
    int train_set_size = 50000, validate_set_size = 10000;

    dataset train_set, validate_set, corrupted_train_set;
    da m, m_corrupt;

    srand(1234);

    load_mnist_dataset(&train_set, &validate_set);

    init_da(&m, 28*28, 500);
    init_da(&m_corrupt, 28*28, 500);

    //train_da(&m, &train_set, &train_set, mini_batch, n_epcho, "da_weight_origin.txt");

    init_dataset(&corrupted_train_set, train_set.N, train_set.nrow, train_set.ncol);
    free(corrupted_train_set.output);
    corrupted_train_set.output = train_set.output;
    corrupt_train_set_input(&train_set, 0.3, &corrupted_train_set);

    train_da(&m_corrupt, &corrupted_train_set, &train_set, mini_batch, n_epcho, "da_weight_corrupt.txt");

    free_da(&m);
    free_dataset(&train_set);
    free_dataset(&validate_set);
    free(corrupted_train_set.input);
}

int main(){
    test_da();
    return 0;
}
