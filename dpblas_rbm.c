#include"dataset.h"
#include"cblas.h"
#include<strings.h>
#include<time.h>

#define MAX_SIZE 1000
#define MAX_BATCH_SIZE 500
#define MAX_STEP 5000
#define eta 0.1

typedef struct rbm{
    int nvisible, nhidden;
    double *W;
    double *b;
    double *c;
} rbm;

double H1[MAX_BATCH_SIZE * MAX_SIZE], H2[MAX_BATCH_SIZE * MAX_SIZE];
double Ph1[MAX_BATCH_SIZE * MAX_SIZE], Ph2[MAX_BATCH_SIZE * MAX_SIZE];
double V2[MAX_BATCH_SIZE * MAX_SIZE];
double Pv1[MAX_BATCH_SIZE * MAX_SIZE], Pv2[MAX_BATCH_SIZE * MAX_SIZE];
double delta_W[MAX_SIZE * MAX_SIZE];
double delta_c[MAX_SIZE], delta_b[MAX_SIZE];
double t1[MAX_SIZE * MAX_SIZE], t2[MAX_SIZE * MAX_SIZE], Ivec[MAX_BATCH_SIZE * MAX_SIZE];

void init_rbm(rbm *m, int nvisible, int nhidden){
    double low, high; 
    int i, j;

    low = -4 * sqrt((double)6 / (nvisible + nhidden));
    high = 4 * sqrt((double)6 / (nvisible + nhidden));

    m->nhidden = nhidden;
    m->nvisible = nvisible;
    m->W = (double*)malloc(nhidden * nvisible * sizeof(double));
    for(i = 0; i < nvisible * nhidden; i++){
        m->W[i] = random_double(low, high);
    }
    m->b = (double*)calloc(nvisible, sizeof(double));
    m->c = (double*)calloc(nhidden, sizeof(double));
}

void free_rbm(rbm *m){
    free(m->W);
    free(m->b);
    free(m->c);
}

void get_hprob(const rbm *m, const double *V, double *Ph, const int batch_size){
    int i;
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->nhidden, m->nvisible,
                1, V, m->nvisible, m->W, m->nvisible,
                0, Ph, m->nhidden);

    cblas_dger(CblasRowMajor, batch_size, m->nhidden,
               1.0, Ivec, 1, m->c, 1, Ph, m->nhidden);

    for(i = 0; i < batch_size * m->nhidden; i++){
        Ph[i] = sigmoid(Ph[i]);
    }
}

void get_hsample(const rbm *m, const double *Ph, double *H, const int batch_size){
    int i;

    for(i = 0; i < batch_size * m->nhidden; i++){
        H[i] = random_double(0, 1) < Ph[i] ? 1 : 0; 
    }
}

void get_vprob(const rbm *m, const double *H, double *Pv, const int batch_size){
    int i;
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, m->nvisible, m->nhidden,
                1, H, m->nhidden, m->W, m->nvisible,
                0, Pv, m->nvisible);

    cblas_dger(CblasRowMajor, batch_size, m->nvisible,
               1.0, Ivec, 1, m->b, 1, Pv, m->nvisible);

    for(i = 0; i < batch_size * m->nvisible; i++){
        Pv[i] = sigmoid(Pv[i]);
    }
}

void get_vsample(const rbm *m, const double *Pv, double *V, const int batch_size){
    int i;

    for(i = 0; i < batch_size * m->nvisible; i++){
        V[i] = random_double(0, 1) < Pv[i] ? 1 : 0; 
    }
}

void gibbs_sample_vhv(const rbm *m, const double *V_start, double *H, double *Ph, 
                      double *V, double *Pv, const int step, const int batch_size){
    int i;

    cblas_dcopy(batch_size * m->nvisible, V_start, 1, V, 1);

    for(i = 0; i < step; i++){
        get_hprob(m, V, Ph, batch_size);
        get_hsample(m, Ph, H, batch_size);
        get_vprob(m, H, Pv, batch_size);
        get_vsample(m, Pv, V, batch_size);
    }

}

void gibbs_sample_hvh(const rbm *m, const double *H_start, double *H, double *Ph, 
                      double *V, double *Pv, const int step, const int batch_size){
    int i;

    cblas_dcopy(batch_size * m->nhidden, H_start, 1, H, 1);

    for(i = 0; i < step; i++){
        get_vprob(m, H, Pv, batch_size);
        get_vsample(m, Pv, V, batch_size);
        get_hprob(m, V, Ph, batch_size);
        get_hsample(m, Ph, H, batch_size);
    }
}

void get_FE(const rbm *m, const double *V, double *FE, const int size){
    int i, j;
     
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, m->nhidden, m->nvisible,
                1, V, m->nvisible, m->W, m->nvisible,
                0, t1, m->nhidden);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                size, m->nhidden, 1,
                1, Ivec, 1, m->c, 1,
                1, t1, m->nhidden);

    for(i = 0; i < size * m->nhidden; i++){
        t1[i] = log(exp(t1[i]) + 1.0);
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, 1, m->nhidden,
                1, t1, m->nhidden, Ivec, 1,
                0, FE, 1);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, 1, m->nvisible,
                -1, V, m->nvisible, m->b, 1,
                -1, FE, 1);
}

double get_PL(const rbm *m, double *V, const int size){
    static int filp_idx = 0;
    int i, j;
    double *FE, *FE_filp, *old_val;
    double PL = 0.0;

    FE = (double*)malloc(size * sizeof(double));
    FE_filp = (double*)malloc(size * sizeof(double));
    old_val = (double*)malloc(size * sizeof(double));

    get_FE(m, V, FE, size);
    for(i = 0; i < size; i++){
        old_val[i] = V[i * m->nvisible + filp_idx];
        V[i * m->nvisible + filp_idx] = 1 - old_val[i];
    }
    get_FE(m, V, FE_filp, size);
    for(i = 0; i < size; i++){
        V[i * m->nvisible + filp_idx] = old_val[i];
    }

    for(i = 0; i < size; i++){
        PL += m->nvisible * log(sigmoid(FE_filp[i] - FE[i]));
    }
    
    free(FE);
    free(FE_filp);
    free(old_val);

    filp_idx = (filp_idx + 1) % m->nvisible;

    return PL / size;
}

void dump_weight(FILE *weight_file, const rbm *m){
    int item_per_line = 28;
    int hidden_unit_count = 100;
    int i, j;

    for(i = 0; i < hidden_unit_count; i++){
        for(j = 0; j < m->nvisible; j++){
            fprintf(weight_file, "%.5lf%s", m->W[i * m->nvisible + j], 
                    ((j+1) % item_per_line == 0) ? "\n" : "\t");
        }
    }
    fflush(weight_file);
}

void dump_sample(FILE *sample_file, const rbm *m, const double *V, const int sample_count){
    int item_per_line = 28;
    int i, j;

    for(i = 0; i < sample_count; i++){
        for(j = 0; j < m->nvisible; j++){
            fprintf(sample_file, "%.5lf%s", V[i * m->nvisible + j], 
                    ((j+1) % item_per_line == 0) ? "\n" : "\t");
        }
    }
    fflush(sample_file);
}

void generate_sample(FILE *sample_file, const rbm *m, double *V_start, const int sample_count){
    double *V_sample;
    int sample_length = 10;
    int i, j;

    V_sample = (double*)malloc(sample_count * m->nvisible * sizeof(double));
    cblas_dcopy(sample_count * m->nvisible, V_start, 1, V_sample, 1);

    for(i = 0; i < sample_length; i++){
        gibbs_sample_vhv(m, V_sample, H2, Ph2, V2, Pv2, 1000, sample_count);
        dump_sample(sample_file, m, Pv2, sample_count);
        cblas_dcopy(sample_count * m->nvisible, V2, 1, V_sample, 1);
    }

    free(V_sample);
}

void dump_rbm(char *rbm_filename, rbm *m){
    int rbm_fd;  
    rio_t rbm_rio; 

    rbm_fd = open(rbm_filename, O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU);
    if(rbm_fd == -1){
        fprintf(stderr, "cannot open %s\n", rbm_filename);
        exit(1);
    }
    rio_readinitb(&rbm_rio, rbm_fd, 1);

    rio_writenb(&rbm_rio, &m->nvisible, sizeof(int));
    rio_writenb(&rbm_rio, &m->nhidden, sizeof(int));

    rio_writenb(&rbm_rio, m->W, m->nhidden * m->nvisible * sizeof(double));
    rio_writenb(&rbm_rio, m->b, m->nvisible * sizeof(double));
    rio_writenb(&rbm_rio, m->c, m->nhidden * sizeof(double));

    close(rbm_fd);
}

void load_rbm(char *rbm_filename, rbm *m){
    int rbm_fd;  
    rio_t rbm_rio; 

    rbm_fd = open(rbm_filename, O_RDONLY);
    if(rbm_fd == -1){
        fprintf(stderr, "cannot open %s\n", rbm_filename);
        exit(1);
    }
    rio_readinitb(&rbm_rio, rbm_fd, 0);

    rio_readnb(&rbm_rio, &m->nvisible, sizeof(int));
    rio_readnb(&rbm_rio, &m->nhidden, sizeof(int));

    rio_readnb(&rbm_rio, m->W, m->nhidden * m->nvisible * sizeof(double));
    rio_readnb(&rbm_rio, m->b, m->nvisible * sizeof(double));
    rio_readnb(&rbm_rio, m->c, m->nhidden * sizeof(double));

    close(rbm_fd);
}

void train_rbm(rbm *m, const dataset_blas *train_set, const dataset_blas *validate_set, 
               const int mini_batch, const int n_epcho, const char *weight_filename){
    int i, j, k, epcho; 
    int batch_count;
    int cd_k = 5;
    double *chain_start, *V1;
    double cost;
    FILE *weight_file;
    time_t start_time, end_time;

    weight_file = fopen(weight_filename, "w");
    if(weight_filename == NULL){
        fprintf(stderr, "no such file %s\n", weight_filename);
        exit(1);
    }
    batch_count = train_set->N / mini_batch;
    chain_start = NULL;

    dump_weight(weight_file, m);

    for(epcho = 0; epcho < n_epcho; epcho++){
        cost = 0;
        start_time = time(NULL);

        for(k = 0; k < batch_count; k++){
#ifdef DEBUG
            if((k+1) % 500 == 0){
                printf("epcho %d batch %d\n", epcho + 1, k + 1);
            }
#endif
            V1 = train_set->input + k * mini_batch * m->nvisible;
            get_hprob(m, V1, Ph1, mini_batch);
            get_hsample(m, Ph1, H1, mini_batch);

            if(chain_start == NULL){
                chain_start = H1;
            }

            gibbs_sample_hvh(m, H1, H2, Ph2, V2, Pv2, cd_k, mini_batch);

            chain_start = H2;

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nhidden, m->nvisible, mini_batch,
                        1.0, Ph2, m->nhidden, V2, m->nvisible,
                        0, t1, m->nvisible);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nhidden, m->nvisible, mini_batch,
                        1.0, Ph1, m->nhidden, V1, m->nvisible,
                        -1, t1, m->nvisible);

            cblas_daxpy(m->nhidden * m->nvisible, eta / mini_batch,
                        t1, 1, m->W, 1);

            cblas_dcopy(m->nvisible * mini_batch, V1, 1, t1, 1);

            cblas_daxpy(m->nvisible * mini_batch, -1, V2, 1, t1, 1);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nvisible, 1, mini_batch,
                        eta / mini_batch, t1, m->nvisible,
                        Ivec, 1, 1, m->b, 1);

            cblas_dcopy(m->nhidden * mini_batch, Ph1, 1, t1, 1);

            cblas_daxpy(m->nhidden * mini_batch, -1, Ph2, 1, t1, 1);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nhidden, 1, mini_batch,
                        eta / mini_batch, t1, m->nvisible,
                        Ivec, 1, 1, m->c, 1);

            cost += get_PL(m, V1, mini_batch);
        }

        end_time = time(NULL);
        printf("epcho %d cost: %.8lf\ttime: %.2lf min\n", epcho+1, cost / batch_count, (double)(end_time - start_time) / 60);

        dump_weight(weight_file, m);
    }

    fclose(weight_file);
}

void test_rbm(){
    int i, j, k, p, q;
    int mini_batch = 20;
    int epcho, n_epcho = 15;
    int train_set_size = 50000, validate_set_size = 10000;
    dataset_blas train_set, validate_set; 
    rbm m;
    FILE *sample_file;

    srand(4321);
    load_mnist_dataset_blas(&train_set, &validate_set);
    init_rbm(&m, 28*28, 500);

    for(i = 0; i < MAX_BATCH_SIZE * MAX_SIZE; i++)
        Ivec[i] = 1.0;

    //load_rbm("rbm_model.dat", &m);

    train_rbm(&m, &train_set, &validate_set, mini_batch, n_epcho, "rbm_weight.txt");
    dump_rbm("rbm_model.dat", &m);

    sample_file = fopen("rbm_sample.txt", "w");
    if(sample_file == NULL){
        fprintf(stderr, "cannot open rbm_sample.txt\n");
        exit(1);
    }
    dump_sample(sample_file, &m, validate_set.input + 100 * m.nvisible, 20);
    generate_sample(sample_file, &m, validate_set.input + 100 * m.nvisible, 20);

    fclose(sample_file);

    free_rbm(&m);
    free_dataset_blas(&validate_set);
    free_dataset_blas(&train_set);
}

int main(){
    test_rbm();
}
