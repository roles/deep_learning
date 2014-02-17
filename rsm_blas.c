#include"dataset.h"
#include"cblas.h"
#include<strings.h>

#define MAXUNIT 2000
#define MAXBATCH_SIZE 100

double v2[MAXUNIT*MAXBATCH_SIZE], h1[MAXUNIT*MAXBATCH_SIZE], h2[MAXUNIT*MAXBATCH_SIZE];
double pv[MAXUNIT*MAXBATCH_SIZE], s[MAXBATCH_SIZE];
double I[MAXUNIT*MAXBATCH_SIZE];
double w_u[MAXUNIT*MAXUNIT], bh_u[MAXUNIT], bv_u[MAXUNIT];
//temporary variable
double a[MAXUNIT*MAXBATCH_SIZE], b[MAXUNIT*MAXBATCH_SIZE];

typedef struct rsm{
    int nvisible, nhidden;
    double *w;
    double *bv, *bh;
} rsm;

void init_rsm(rsm *m, int nvisible, int nhidden){
    double low, high; 
    int i, j;

    low = -4 * sqrt((double)6 / (nvisible + nhidden));
    high = 4 * sqrt((double)6 / (nvisible + nhidden));

    m->nhidden = nhidden;
    m->nvisible = nvisible;
    m->w = (double*)malloc(nhidden * nvisible * sizeof(double));
    for(i = 0; i < nvisible * nhidden; i++){
        m->w[i] = random_double(low, high);
    }
    m->bv = (double*)calloc(nvisible, sizeof(double));
    m->bh = (double*)calloc(nhidden, sizeof(double));
}

void free_rsm(rsm *m){
    free(m->w);
    free(m->bv);
    free(m->bh);
}

void get_hidden(rsm *m, double *v, double *h, double *s, int batch_size){
    int i;
    double *t;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, m->nhidden, m->nvisible,
                1, v, m->nvisible, m->w, m->nhidden,
                0, h, m->nhidden);

    if(s != NULL)
        t = s;
    else
        t = a;
    //sum each v[i] in a batch
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, 1, m->nvisible,
                1, v, m->nvisible, I, 1,
                0, t, 1);
     
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->nhidden, 1,
                1, t, 1, m->bh, 1,
                1, h, m->nhidden);

    for(i = 0; i < batch_size * m->nhidden; i++){
        h[i] = sigmoid(h[i]); 
    }
}

void get_visible_prob(rsm *m, double *h, double *pv, int batch_size){
    int i;
    double sum;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->nvisible, m->nhidden,
                1, h, m->nhidden, m->w, m->nhidden,
                0, pv, m->nvisible);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                batch_size, m->nvisible, 1,
                1, I, 1, m->bv, 1,
                1, pv, m->nvisible);

    for(i = 0; i < batch_size * m->nvisible; i++){
        pv[i] = exp(pv[i]);
    }

    //sum
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batch_size, 1, m->nvisible,
                1, pv, m->nvisible, I, 1,
                0, a, 1);

    for(i = 0; i < batch_size; i++){
        cblas_dscal(m->nvisible, 1.0 / a[i], pv + i * m->nvisible, 1);
        //printf("sum:%.2lf\n", cblas_dasum(m->nvisible, pv + i * m->nvisible, 1));
    }

}

void sample_visible(rsm *m, double *pv, double *s, double *v, int batch_size){
    int i, j;
    int *o;
    float *pv_f;

    pv_f = (float*)malloc(batch_size * m->nvisible * sizeof(float));
    for(i = 0; i < batch_size * m->nvisible; i++)
        pv_f[i] = pv[i];
    
    for(i = 0; i < batch_size; i++){
        o = genmul(s[i], pv_f + m->nvisible * i, m->nvisible);
        for(j = 0; j < m->nvisible; j++){
            v[m->nvisible * i + j] = o[j];
        }
        free(o);
    }
    free(pv_f);
}

void cdn(rsm *m, double *v1, double *h1, double *v2, double *h2,
         int batch_size){

}

double get_likelihood(rsm *m, double *v1, double *pv, int batch_size){
    int i;
    double lik;
    
    for(i = 0; i < batch_size * m->nvisible; i++)
        a[i] = log(pv[i]); 
    lik = cblas_ddot(batch_size * m->nvisible,
                     v1, 1, a, 1);
    return(lik);
}

void dump_model(rsm *m, char *filename){
    rio_t rio_model;
    int fd_model;

    if((fd_model = open(filename, O_WRONLY | O_TRUNC | O_CREAT, S_IRWXU)) == -1){
        printf("open %s failed\n", filename);
    }
    rio_readinitb(&rio_model, fd_model, 1);
    rio_writenb(&rio_model, &m->nvisible, sizeof(int));
    rio_writenb(&rio_model, &m->nhidden, sizeof(int));
    rio_writenb(&rio_model, m->w, m->nvisible * m->nhidden * sizeof(double));
    rio_writenb(&rio_model, m->bv, m->nvisible * sizeof(double));
    rio_writenb(&rio_model, m->bh, m->nhidden * sizeof(double));
    close(fd_model);
}

void load_model(rsm *m, char *filename){
    rio_t rio_model;
    int fd_model;

    if((fd_model = open(filename, O_RDONLY)) == -1){
        printf("open %s failed\n", filename);
    }
    rio_readinitb(&rio_model, fd_model, 0);
    rio_readnb(&rio_model, &m->nvisible, sizeof(int));
    rio_readnb(&rio_model, &m->nhidden, sizeof(int));
    m->w = (double*)malloc(m->nhidden * m->nvisible * sizeof(double));
    m->bv = (double*)malloc(m->nvisible * sizeof(double));
    m->bh = (double*)malloc(m->nhidden * sizeof(double));
    rio_readnb(&rio_model, m->w, m->nvisible * m->nhidden * sizeof(double));
    rio_readnb(&rio_model, m->bv, m->nvisible * sizeof(double));
    rio_readnb(&rio_model, m->bh, m->nhidden * sizeof(double));
    close(fd_model);
}

void print_prob(rsm *m, dataset_blas *train_set, char *filename){
    int niter, batch_size;
    int minibatch = 10;
    FILE *f;
    double *v1;
    int i, j, k;

    if((f = fopen(filename, "w+")) == NULL){
        printf("open %s failed\n", filename);
    }
    niter = (train_set->N-1)/minibatch + 1;
    for(k = 0; k < niter; k++){
        if(k == niter - 1){
            batch_size = train_set->N - minibatch * (niter-1);
        }else{
            batch_size = minibatch;
        }
        v1 = train_set->input + train_set->n_feature * minibatch * k;
        get_hidden(m, v1, h1, s, batch_size);            
        get_visible_prob(m, h1, pv, batch_size);

        for(i = 0; i < batch_size; i++){
            for(j = 0; j < m->nvisible; j++){
                fprintf(f, "%.5lf%s", pv[i * m->nvisible + j],
                        (j == (m->nvisible-1)) ? "\n" : "\t");
            }
        }
    }   

    fclose(f);
}

void train_rsm(dataset_blas *train_set, int nvisible, int nhidden, int epoch, double lr,
               int minibatch, double momentum, char *model_file){
    rsm m;
    int i, j, k;
    int nepoch;
    double *v1;
    int batch_size, niter;
    int wc;
    double lik;
    double delta;
    int *idx;

    init_rsm(&m, nvisible, nhidden);

    wc = (int)cblas_dasum(train_set->N * train_set->n_feature, train_set->input, 1);
    v1 = (double*)malloc(minibatch * nvisible * sizeof(double));
    idx = (int*)malloc(train_set->N * sizeof(int));
    for(i = 0; i < train_set->N; i++)
        idx[i] = i;
    bzero(w_u, sizeof(w_u));
    bzero(bh_u, sizeof(bh_u));
    bzero(bv_u, sizeof(bv_u));
    
    for(nepoch = 0; nepoch < epoch; nepoch++){
        
        niter = (train_set->N-1)/minibatch + 1;
        lik = 0;
        delta = lr / (1.0*niter);
        shuffle(idx, train_set->N);

        for(k = 0; k < niter; k++){
            if(k == niter - 1){
                batch_size = train_set->N - minibatch * (niter-1);
            }else{
                batch_size = minibatch;
            }
            //v1 = train_set->input + train_set->n_feature * minibatch * k;
            for(i = 0; i < batch_size; i++){
                cblas_dcopy(m.nvisible, train_set->input + m.nvisible * idx[k*minibatch+i],
                            1, v1 + m.nvisible * i, 1);
            }
            get_hidden(&m, v1, h1, s, batch_size);            
            get_visible_prob(&m, h1, pv, batch_size);
            sample_visible(&m, pv, s, v2, batch_size);
            get_hidden(&m, v2, h2, NULL, batch_size);
            
            lik += get_likelihood(&m, v1, pv, batch_size);
            
            //update w_u
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nvisible, m.nhidden, batch_size,
                        1, v2, m.nvisible, h2, m.nhidden,
                        0, a, m.nhidden);
            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m.nvisible, m.nhidden, batch_size,
                        1, v1, m.nvisible, h1, m.nhidden,
                        -1, a, m.nhidden);
            cblas_daxpy(m.nvisible * m.nhidden, momentum, w_u, 1,
                        a, 1);
            cblas_dcopy(m.nvisible * m.nhidden, a, 1, w_u, 1);

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

            cblas_daxpy(m.nvisible * m.nhidden, delta, w_u, 1,
                        m.w, 1);
            cblas_daxpy(m.nvisible, delta, bv_u, 1,
                        m.bv, 1);
            cblas_daxpy(m.nhidden, delta, bh_u, 1,
                        m.bh, 1);
        }

        printf("[epoch %d] ppl:%.2lf\n", nepoch + 1, exp(-lik / wc));
    }
    dump_model(&m, model_file);
    print_prob(&m, train_set, "../data/rsm/test.prob");

    free(v1);
    free_rsm(&m);
}

void test_rsm(){
    dataset_blas train_set;

    int nhidden = 50;
    int epoch = 1000;
    double lr = 0.001;
    int minibatch = 10;
    double momentum = 0.9;

    //load_corpus("../data/tcga.train", &train_set);
    load_corpus("../data/train", &train_set);
    train_rsm(&train_set, train_set.n_feature, nhidden, epoch, lr,
              minibatch, momentum, "../data/rsm/test.model");
}

int init(){
    srand(1234);
    int i;
    for(i = 0; i < MAXUNIT*MAXBATCH_SIZE; i++)
        I[i] = 1;
}

int main(){
    init();
    test_rsm();
}
