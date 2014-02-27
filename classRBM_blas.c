#include"dataset.h"
#include"cblas.h"
#include<strings.h>

#define MAXUNIT 5500
#define MAXBATCH_SIZE 100

double v2[MAXUNIT*MAXBATCH_SIZE], h1[MAXUNIT*MAXBATCH_SIZE], h2[MAXUNIT*MAXBATCH_SIZE];
double y2[MAXUNIT*MAXBATCH_SIZE];
double ph1[MAXUNIT*MAXBATCH_SIZE], ph2[MAXUNIT*MAXBATCH_SIZE], pv[MAXUNIT*MAXBATCH_SIZE], py[MAXUNIT*MAXBATCH_SIZE];
double I[MAXUNIT*MAXBATCH_SIZE];
double w_u[MAXUNIT*MAXUNIT], u_u[MAXUNIT*MAXBATCH_SIZE];
double by_u[MAXUNIT], bh_u[MAXUNIT], bv_u[MAXUNIT];
//temporary variable
double a[MAXUNIT*MAXUNIT], b[MAXUNIT*MAXUNIT];

typedef struct crbm{
    int nvisible, nhidden, ncat;
    double *w;
    double *u;
    double *bv, *bh, *by;
} crbm;

void init_crbm(crbm *m, int nvisible, int nhidden, int ncat){
    double low, high; 
    int i, j;

    low = -4 * sqrt((double)6 / (nvisible + nhidden));
    high = 4 * sqrt((double)6 / (nvisible + nhidden));

    m->nhidden = nhidden;
    m->nvisible = nvisible;
    m->ncat = ncat;
    m->w = (double*)malloc(nhidden * nvisible * sizeof(double));
    m->u = (double*)malloc(nhidden * ncat * sizeof(double));
    for(i = 0; i < nvisible * nhidden; i++){
        m->w[i] = random_double(low, high);
    }
    for(i = 0; i < ncat * nhidden; i++){
        m->u[i] = random_double(low, high);
    }
    m->bv = (double*)calloc(nvisible, sizeof(double));
    m->bh = (double*)calloc(nhidden, sizeof(double));
    m->by = (double*)calloc(ncat, sizeof(double));
}

void free_crbm(crbm *m){
    free(m->w);
    free(m->u);
    free(m->bv);
    free(m->bh);
    free(m->by);
}

void get_hidden(crbm *m, double *v, double *y, double *ph, int batch_size){
    int i;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->nhidden, m->nvisible,
                1, v, m->nvisible, m->w, m->nvisible,
                0, ph, m->nhidden);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->nhidden, m->ncat,
                1, y, m->ncat, m->u, m->ncat,
                1, ph, m->nhidden);

    cblas_dger(CblasRowMajor, batch_size, m->nhidden, 1,
               I, 1, m->bh, 1, ph, m->nhidden);

    for(i = 0; i < batch_size * m->nhidden; i++){
        ph[i] = sigmoid(ph[i]); 
    }
}

void sample_hidden(crbm *m, double *ph, double *h, int batch_size){
    int i;

    for(i = 0; i < m->nhidden * batch_size; i++){
        h[i] = random_double(0, 1) < ph[i] ? 1 : 0; 
    }
}


void get_visible(crbm *m, double *h, double *pv, int batch_size){
    int i;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, m->nvisible, m->nhidden,
                1.0, h, m->nhidden, m->w, m->nvisible,
                0, pv, m->nvisible);

    cblas_dger(CblasRowMajor, batch_size, m->nvisible, 1,
               I, 1, m->bv, 1, pv, m->nvisible);

    for(i = 0; i < batch_size * m->nvisible; i++){
        pv[i] = sigmoid(pv[i]); 
    }
}

void sample_visible(crbm *m, double *pv, double *v, int batch_size){
    int i;

    for(i = 0; i < m->nvisible * batch_size; i++){
        v[i] = random_double(0, 1) < pv[i] ? 1 : 0; 
    }
}

void get_class(crbm *m, double *h, double *py, int batch_size){
    int i;
    double sum;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, m->ncat, m->nhidden,
                1.0, h, m->nhidden, m->u, m->ncat,
                0, py, m->ncat);

    cblas_dger(CblasRowMajor, batch_size, m->ncat, 1,
               I, 1, m->by, 1, py, m->ncat);

    for(i = 0; i < batch_size * m->ncat; i++){
        py[i] = exp(py[i]);
    }

    //sum
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, 1, m->ncat,
                1, py, m->ncat, I, 1,
                0, a, 1);

    for(i = 0; i < batch_size; i++){
        cblas_dscal(m->ncat, 1.0 / a[i], py + i * m->ncat, 1);
        //printf("sum:%.2lf\n", cblas_dasum(m->ncat, py + i * m->ncat, 1));
    }
}

/*void sample_class(crbm *m, double *py, double *y, int batch_size){
    int i, j;
    int *o;
    float *py_f;

    py_f = (float*)malloc(batch_size * m->ncat * sizeof(float));
    for(i = 0; i < batch_size * m->ncat; i++)
        py_f[i] = py[i];
    
    for(i = 0; i < batch_size; i++){
        o = genmul(1, py_f + m->ncat * i, m->ncat);
        for(j = 0; j < m->ncat; j++){
            y[m->ncat * i + j] = o[j];
        }
        free(o);
    }
    free(py_f);
}*/

void sample_class(crbm *m, double *py, double *y, int batch_size){
    int i, j;
    double sum, u;
    
    bzero(y, m->ncat * batch_size * sizeof(double));
    for(i = 0; i < batch_size; i++){
        sum = 0; 
        u = random_double(0, 1);
        j = 0;
        while(j < m->ncat){
            sum += py[m->ncat*i + j];
            if(sum >= u){
                y[m->ncat * i + j] = 1;
                break;
            }
            j++;
        }
    }
}

double get_likelihood(crbm *m, double *v1, double *pv, int batch_size){
    int i;
    double lik = 0;
    
    for(i = 0; i < batch_size * m->nvisible; i++)
        a[i] = log(pv[i] + 0.01); 
    lik = cblas_ddot(m->nvisible * batch_size, v1, 1, 
                      a, 1);
    /*
    for(i = 0; i < batch_size; i++){
        lik = cblas_ddot(m->nvisible, v1 + m->nvisible * i, 1, 
                          a + m->nvisible * i, 1);

        printf("lik:%.5lf\n", lik);
    }*/
    return(lik);
}

int get_error(crbm *m, double *y, uint8_t *label, int batch_size){
    int err = 0;
    int i;

    for(i = 0; i < batch_size; i++){
        if(y[m->ncat * i + label[i]] < 0.99){
            err++;
        }
    }
    return(err);
}

void dump_model(crbm *m, char *filename){
    rio_t rio_model;
    int fd_model;

    if((fd_model = open(filename, O_WRONLY | O_TRUNC | O_CREAT, S_IRWXU)) == -1){
        printf("open %s failed\n", filename);
    }
    rio_readinitb(&rio_model, fd_model, 1);
    rio_writenb(&rio_model, &m->nvisible, sizeof(int));
    rio_writenb(&rio_model, &m->nhidden, sizeof(int));
    rio_writenb(&rio_model, &m->ncat, sizeof(int));
    rio_writenb(&rio_model, m->w, m->nvisible * m->nhidden * sizeof(double));
    rio_writenb(&rio_model, m->u, m->ncat * m->nhidden * sizeof(double));
    rio_writenb(&rio_model, m->bv, m->nvisible * sizeof(double));
    rio_writenb(&rio_model, m->bh, m->nhidden * sizeof(double));
    rio_writenb(&rio_model, m->by, m->ncat * sizeof(double));
    close(fd_model);
}

void load_model(crbm *m, char *filename){
    rio_t rio_model;
    int fd_model;

    if((fd_model = open(filename, O_RDONLY)) == -1){
        printf("open %s failed\n", filename);
    }
    rio_readinitb(&rio_model, fd_model, 0);
    rio_readnb(&rio_model, &m->nvisible, sizeof(int));
    rio_readnb(&rio_model, &m->nhidden, sizeof(int));
    rio_readnb(&rio_model, &m->ncat, sizeof(int));
    m->w = (double*)malloc(m->nhidden * m->nvisible * sizeof(double));
    m->u = (double*)malloc(m->nhidden * m->ncat * sizeof(double));
    m->bv = (double*)malloc(m->nvisible * sizeof(double));
    m->bh = (double*)malloc(m->nhidden * sizeof(double));
    m->by = (double*)malloc(m->ncat * sizeof(double));
    rio_readnb(&rio_model, m->w, m->nvisible * m->nhidden * sizeof(double));
    rio_readnb(&rio_model, m->u, m->ncat * m->nhidden * sizeof(double));
    rio_readnb(&rio_model, m->bv, m->nvisible * sizeof(double));
    rio_readnb(&rio_model, m->bh, m->nhidden * sizeof(double));
    rio_readnb(&rio_model, m->by, m->ncat * sizeof(double));
    close(fd_model);
}

void train_crbm(dataset_blas *train_set, int nvisible, int nhidden, int nlabel,
               int epoch, double lr,
               int minibatch, double momentum, char *model_file){
    crbm m;
    int i, j, k;
    int nepoch;
    double *v1, *y1;
    uint8_t *l;
    int batch_size, niter;
    int wc;
    int err;
    double lik;
    double delta;
    int *idx;
    time_t start_t, end_t;

    init_crbm(&m, nvisible, nhidden, nlabel);

    //wc = (int)cblas_dasum(train_set->N * train_set->n_feature, train_set->input, 1);
    niter = (train_set->N-1)/minibatch + 1;
    delta = lr / (1.0*minibatch);
    /*
     * shuffle training data
    v1 = (double*)malloc(minibatch * m.nvisible * sizeof(double));
    y1 = (double*)malloc(minibatch * m.ncat * sizeof(double));
    l = (uint8_t*)malloc(minibatch * sizeof(uint8_t));
    idx = (int*)malloc(train_set->N * sizeof(int));
    for(i = 0; i < train_set->N; i++)
        idx[i] = i;*/
    bzero(w_u, sizeof(w_u));
    bzero(u_u, sizeof(u_u));
    bzero(bh_u, sizeof(bh_u));
    bzero(bv_u, sizeof(bv_u));
    bzero(by_u, sizeof(by_u));

    //shuffle(idx, train_set->N);
    
    for(nepoch = 0; nepoch < epoch; nepoch++){
        
        lik = 0;
        err = 0;
        start_t = time(NULL);

        for(k = 0; k < niter; k++){
#ifdef DEBUG
            if((k+1) % 200 == 0){
                printf("batch %d\n", k+1);
            }
#endif
            if(k == niter - 1){
                batch_size = train_set->N - minibatch * (niter-1);
            }else{
                batch_size = minibatch;
            }

            v1 = train_set->input + train_set->n_feature * minibatch * k;
            y1 = train_set->label + train_set->nlabel * minibatch * k;
            l = train_set->output + minibatch * k;
            /*
             * shuffle training data
            for(i = 0; i < batch_size; i++){
                cblas_dcopy(m.nvisible, train_set->input + m.nvisible * idx[k*minibatch+i],
                            1, v1 + m.nvisible * i, 1);
                cblas_dcopy(m.ncat, train_set->label + m.ncat * idx[k*minibatch+i],
                            1, y1 + m.ncat * i, 1);
                l[i] = train_set->output[idx[k*minibatch+i]];
            }*/
            get_hidden(&m, v1, y1, ph1, batch_size);            
            sample_hidden(&m, ph1, h1, batch_size);

            get_visible(&m, h1, pv, batch_size);
            sample_visible(&m, pv, v2, batch_size);

            get_class(&m, h1, py, batch_size);
            sample_class(&m, py, y2, batch_size);

            get_hidden(&m, v2, y2, ph2, batch_size);
            sample_hidden(&m, ph2, h2, batch_size);
            
            //lik += get_likelihood(&m, v1, pv, batch_size);
            err += get_error(&m, y2, l, batch_size);
            
            //update w_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nhidden, m.nvisible, batch_size,
                        1.0, ph2, m.nhidden, v2, m.nvisible,
                        0, a, m.nvisible);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nhidden, m.nvisible, batch_size,
                        1.0, ph1, m.nhidden, v1, m.nvisible,
                        -1, a, m.nvisible);
            cblas_daxpy(m.nvisible * m.nhidden, momentum, w_u, 1,
                        a, 1);
            cblas_dcopy(m.nvisible * m.nhidden, a, 1, w_u, 1);

            //update u_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nhidden, m.ncat, batch_size,
                        1.0, ph2, m.nhidden, y2, m.ncat,
                        0, a, m.ncat);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nhidden, m.ncat, batch_size,
                        1.0, ph1, m.nhidden, y1, m.ncat,
                        -1, a, m.ncat);
            cblas_daxpy(m.ncat * m.nhidden, momentum, u_u, 1,
                        a, 1);
            cblas_dcopy(m.ncat * m.nhidden, a, 1, u_u, 1);

            //update bv_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nvisible, 1, batch_size,
                        1.0, v2, m.nvisible, I, 1,
                        0, a, 1);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nvisible, 1, batch_size,
                        1.0, v1, m.nvisible, I, 1,
                        -1, a, 1);
            cblas_daxpy(m.nvisible, momentum, bv_u, 1,
                        a, 1);
            cblas_dcopy(m.nvisible, a, 1, bv_u, 1);

            //update by_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.ncat, 1, batch_size,
                        1.0, y2, m.ncat, I, 1,
                        0, a, 1);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.ncat, 1, batch_size,
                        1.0, y1, m.ncat, I, 1,
                        -1, a, 1);
            cblas_daxpy(m.ncat, momentum, by_u, 1,
                        a, 1);
            cblas_dcopy(m.ncat, a, 1, by_u, 1);

            //update bh_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nhidden, 1, batch_size,
                        1.0, ph2, m.nhidden, I, 1,
                        0, a, 1);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nhidden, 1, batch_size,
                        1.0, ph1, m.nhidden, I, 1,
                        -1, a, 1);
            cblas_daxpy(m.nhidden, momentum, bh_u, 1,
                        a, 1);
            cblas_dcopy(m.nhidden, a, 1, bh_u, 1);

            //change parameter
            cblas_daxpy(m.nvisible * m.nhidden, delta, w_u, 1,
                        m.w, 1);
            cblas_daxpy(m.ncat * m.nhidden, delta, u_u, 1,
                        m.u, 1);
            cblas_daxpy(m.nvisible, delta, bv_u, 1,
                        m.bv, 1);
            cblas_daxpy(m.ncat, delta, by_u, 1,
                        m.by, 1);
            cblas_daxpy(m.nhidden, delta, bh_u, 1,
                        m.bh, 1);
        }
        end_t = time(NULL);

        printf("[epoch %d] error:%.5lf%%\ttime:%.2fmin\n", nepoch + 1, 
               err * 100.0 / train_set->N, (end_t - start_t) / 60.0);
    }
    dump_model(&m, model_file);
    //print_prob(&m, train_set, "../data/rsm/test.prob");

    /*
     * shuffle training data
    free(v1);
    free(y1);
    free(l);*/
    free_crbm(&m);
}

void test_crbm(){
    dataset_blas train_set, valid_set;

    int nhidden = 500;
    int epoch = 30;
    double lr = 0.1;
    int minibatch = 10;
    double momentum = 0;

    //load_corpus("../data/20newsgroup/train.data.format", &train_set);
    //load_corpus_label("../data/20newsgroup/train.label.format", &train_set);
    load_corpus("../data/tcga/train.pm.data", &train_set);
    load_corpus_label("../data/tcga/train.pm.label", &train_set);
    //load_mnist_dataset_blas(&train_set, &valid_set);
    train_crbm(&train_set, train_set.n_feature, nhidden, train_set.nlabel, epoch, lr,
               minibatch, momentum, "../data/tcga/crbm.pm.model");

    free_dataset_blas(&train_set);
}

void analyse(){
    crbm m;
    double* u[MAXUNIT*20];
    int i, j;
    FILE *u_file, *w_file;
    
    load_model(&m, "../data/tcga/crbm.pm.model");
    if((u_file = fopen("../data/tcga/crbm.pm.model.u", "w+")) == NULL){
        printf("open %s failed\n", "crbm.pm.model.u");
        exit(1);
    }   
    if((w_file = fopen("../data/tcga/crbm.pm.model.w", "w+")) == NULL){
        printf("open %s failed\n", "crbm.pm.model.w");
        exit(1);
    }   
    for(i = 0; i < m.nhidden; i++){
        for(j = 0; j < m.ncat; j++){
            fprintf(u_file, "%.5lf%s", m.u[i*m.ncat + j], (j == m.ncat-1) ? "\n" : "\t");
        }
    }
    for(i = 0; i < m.nhidden; i++){
        for(j = 0; j < m.nvisible; j++){
            fprintf(w_file, "%.5lf%s", m.w[i*m.nvisible + j], (j == m.nvisible-1) ? "\n" : "\t");
        }
    }
    fclose(u_file);
    fclose(w_file);
}

int init(){
    srand(1234);
    int i;
    for(i = 0; i < MAXUNIT*MAXBATCH_SIZE; i++)
        I[i] = 1;
}

int main(){
    init();
    //test_crbm();
    analyse();
}
