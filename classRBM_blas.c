#include"dataset.h"
#include"cblas.h"
#include<strings.h>

#define MAXUNIT 5000
#define MAXBATCH_SIZE 100

double v2[MAXUNIT*MAXBATCH_SIZE], h1[MAXUNIT*MAXBATCH_SIZE], h2[MAXUNIT*MAXBATCH_SIZE];
double y2[MAXUNIT*MAXBATCH_SIZE];
double ph1[MAXUNIT*MAXBATCH_SIZE], ph2[MAXUNIT*MAXBATCH_SIZE], pv[MAXUNIT*MAXBATCH_SIZE], py[MAXUNIT*MAXBATCH_SIZE];
double I[MAXUNIT*MAXBATCH_SIZE];
double w_u[MAXUNIT*MAXUNIT], u_u[MAXUNIT*MAXBATCH_SIZE];
double by_u[MAXUNIT], bh_u[MAXUNIT], bv_u[MAXUNIT];
//temporary variable
double a[MAXUNIT*MAXBATCH_SIZE], b[MAXUNIT*MAXBATCH_SIZE];

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

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, m->nhidden, m->nvisible,
                1, v, m->nvisible, m->w, m->nhidden,
                0, ph, m->nhidden);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, m->nhidden, m->ncat,
                1, y, m->ncat, m->u, m->nhidden,
                1, ph, m->nhidden);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->nhidden, 1,
                1, I, 1, m->bh, 1,
                1, ph, m->nhidden);

    for(i = 0; i < batch_size * m->nhidden; i++){
        ph[i] = sigmoid(ph[i]); 
    }
}

void sample_hidden(crbm *m, double *ph, double *h, int batch_size){
    int i;
    double u;

    for(i = 0; i < m->nhidden * batch_size; i++){
        u = random_double(0,1);    
        if(u <= ph[i])
            h[i] = 1;
        else
            h[i] = 0;
    }
}


void get_visible(crbm *m, double *h, double *pv, int batch_size){
    int i;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->nhidden, m->nvisible,
                1, h, m->nhidden, m->w, m->nhidden,
                0, pv, m->nhidden);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->nvisible, 1,
                1, I, 1, m->bv, 1,
                1, pv, m->nvisible);

    for(i = 0; i < batch_size * m->nhidden; i++){
        pv[i] = sigmoid(pv[i]); 
    }
}

void sample_visible(crbm *m, double *pv, double *v, int batch_size){
    int i;
    double u;

    for(i = 0; i < m->nhidden * batch_size; i++){
        u = random_double(0,1);    
        if(u <= pv[i])
            v[i] = 1;
        else
            v[i] = 0;
    }
}

void get_class(crbm *m, double *h, double *py, int batch_size){
    int i;
    double sum;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->ncat, m->nhidden,
                1, h, m->nhidden, m->u, m->nhidden,
                0, py, m->ncat);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->ncat, 1,
                1, I, 1, m->by, 1,
                1, pv, m->ncat);

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
        //printf("sum:%.2lf\n", cblas_dasum(m->nvisible, pv + i * m->nvisible, 1));
    }
}

void sample_class(crbm *m, double *py, double *y, int batch_size){
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
}

double get_likelihood(crbm *m, double *v1, double *pv, int batch_size){
    int i;
    double lik;
    
    for(i = 0; i < batch_size * m->nvisible; i++)
        a[i] = log(pv[i]); 
    lik = cblas_ddot(batch_size * m->nvisible,
                     v1, 1, a, 1);
    return(lik);
}

void train_crbm(dataset_blas *train_set, int nvisible, int nhidden, int nlabel,
               int epoch, double lr,
               int minibatch, double momentum, char *model_file){
    crbm m;
    int i, j, k;
    int nepoch;
    double *v1, *y1;
    int batch_size, niter;
    int wc;
    double lik;
    double delta;
    int *idx;

    init_crbm(&m, nvisible, nhidden, nlabel);

    wc = (int)cblas_dasum(train_set->N * train_set->n_feature, train_set->input, 1);
    niter = (train_set->N-1)/minibatch + 1;
    delta = lr / (1.0*niter);
    /*
     * shuffle training data
    v1 = (double*)malloc(minibatch * nvisible * sizeof(double));
    y1 = (double*)malloc(minibatch * ncat * sizeof(double));
    idx = (int*)malloc(train_set->N * sizeof(int));
    for(i = 0; i < train_set->N; i++)
        idx[i] = i;
    */
    bzero(w_u, sizeof(w_u));
    bzero(u_u, sizeof(w_u));
    bzero(bh_u, sizeof(bh_u));
    bzero(bv_u, sizeof(bv_u));
    bzero(by_u, sizeof(bv_u));
    
    for(nepoch = 0; nepoch < epoch; nepoch++){
        
        shuffle(idx, train_set->N);
        lik = 0;

        for(k = 0; k < niter; k++){
            if(k == niter - 1){
                batch_size = train_set->N - minibatch * (niter-1);
            }else{
                batch_size = minibatch;
            }
            v1 = train_set->input + train_set->n_feature * minibatch * k;
            y1 = train_set->label + train_set->nlabel * minibatch * k;
            /*
             * shuffle training data
            for(i = 0; i < batch_size; i++){
                cblas_dcopy(m.nvisible, train_set->input + m.nvisible * idx[k*minibatch+i],
                            1, v1 + m.nvisible * i, 1);
                cblas_dcopy(m.ncat, train_set->label + m.ncat * idx[k*minibatch+i],
                            1, y1 + m.ncat * i, 1);
            }
            */
            get_hidden(&m, v1, y1, ph1, batch_size);            
            sample_visible(&m, ph1, h1, batch_size);

            get_visible(&m, h1, pv, batch_size);
            sample_visible(&m, pv, v2, batch_size);

            get_class(&m, h1, py, batch_size);
            sample_class(&m, py, y2, batch_size);

            get_hidden(&m, v2, y2, ph2, batch_size);
            
            lik += get_likelihood(&m, v1, pv, batch_size);
            
            //update w_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nvisible, m.nhidden, batch_size,
                        1, v2, m.nvisible, ph2, m.nhidden,
                        0, a, m.nhidden);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nvisible, m.nhidden, batch_size,
                        1, v1, m.nvisible, ph1, m.nhidden,
                        -1, a, m.nhidden);
            cblas_daxpy(m.nvisible * m.nhidden, momentum, w_u, 1,
                        a, 1);
            cblas_dcopy(m.nvisible * m.nhidden, a, 1, w_u, 1);

            //update u_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.ncat, m.nhidden, batch_size,
                        1, y2, m.ncat, ph2, m.nhidden,
                        0, a, m.nhidden);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.ncat, m.nhidden, batch_size,
                        1, y1, m.ncat, ph1, m.nhidden,
                        -1, a, m.nhidden);
            cblas_daxpy(m.ncat * m.nhidden, momentum, u_u, 1,
                        a, 1);
            cblas_dcopy(m.ncat * m.nhidden, a, 1, u_u, 1);

            //update bv_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nvisible, 1, batch_size,
                        1, v2, m.nvisible, I, 1,
                        0, a, 1);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nvisible, 1, batch_size,
                        1, v1, m.nvisible, I, 1,
                        -1, a, 1);
            cblas_daxpy(m.nvisible, momentum, bv_u, 1,
                        a, 1);
            cblas_dcopy(m.nvisible, a, 1, bv_u, 1);

            //update by_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.ncat, 1, batch_size,
                        1, y2, m.ncat, I, 1,
                        0, a, 1);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.ncat, 1, batch_size,
                        1, y1, m.ncat, I, 1,
                        -1, a, 1);
            cblas_daxpy(m.ncat, momentum, by_u, 1,
                        a, 1);
            cblas_dcopy(m.ncat, a, 1, bv_u, 1);

            //update bh_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nhidden, 1, batch_size,
                        1, h2, m.nhidden, I, 1,
                        0, a, 1);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nhidden, 1, batch_size,
                        1, h1, m.nhidden, I, 1,
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

        printf("[epoch %d] ppl:%.2lf\n", nepoch + 1, exp(-lik / wc));
    }
    //dump_model(&m, model_file);
    //print_prob(&m, train_set, "../data/rsm/test.prob");

    /*
     * shuffle training data
    free(v1);
    free(y1);
    */
    free_crbm(&m);
}

void test_crbm(){
    dataset_blas train_set;

    int nhidden = 1000;
    int epoch = 15;
    double lr = 0.001;
    int minibatch = 100;
    double momentum = 0.9;

    //load_corpus("../data/tcga.train", &train_set);
    load_corpus("../data/20newsgroup/train.format", &train_set);
    load_corpus_label("../data/20newsgroup/train.label", &train_set);
    train_crbm(&train_set, train_set.n_feature, train_set.nlabel, nhidden, epoch, lr,
               minibatch, momentum, "../data/rsm/test.model");

    free_dataset_blas(&train_set);
}

int init(){
    srand(1234);
    int i;
    for(i = 0; i < MAXUNIT*MAXBATCH_SIZE; i++)
        I[i] = 1;
}

int main(){
    init();
    test_crbm();
}
