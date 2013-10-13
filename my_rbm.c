#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<stdint.h>
#include<arpa/inet.h>
#include<stdlib.h>
#include<math.h>

#define DEFAULT_MAXSIZE 1000
#define eta 0.1

typedef struct dataset{
    uint32_t N;
    uint32_t nrow, ncol;
    double **input;
} dataset;

typedef struct rbm {
    int nvisible, nhidden;
    double **W;
    double *hbias, *vbias;
} rbm;

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

void init_model(rbm *m, int nvisible, int nhidden){
    double low, high; 
    int i, j;

    low = -4 * sqrt((double)6 / (nvisible + nhidden));
    high = 4 * sqrt((double)6 / (nvisible + nhidden));

    m->nhidden = nhidden;
    m->nvisible = nvisible;
    m->W = (double**)malloc(nhidden * sizeof(double*));
    for(i = 0; i < nhidden; i++){
        m->W[i] = (double*)malloc(nvisible * sizeof(double));
    }
    for(i = 0; i < nhidden; i++){
        for(j = 0; j < nvisible; j++){
            m->W[i][j] = random_double(low, high);
        }
    }
    m->vbias = (double*)calloc(nvisible, sizeof(double));
    m->hbias = (double*)calloc(nvisible, sizeof(double));
}

void free_model(rbm *m){
    int i;

    for(i = 0; i < m->nhidden; i++){
        free(m->W[i]);
    }
    free(m->W);
    free(m->vbias);
    free(m->hbias);

}

void get_hprob_given_vsample(const rbm *m, const double *vsample, double *hprob){
    int i, j;
    double s = 0;
    for(i = 0; i < m->nhidden; i++){
        s = 0;
        for(j = 0; j < m->nvisible; j++){
            s += m->W[i][j] * vsample[j]; 
        }
        hprob[i] = sigmoid(m->hbias[i] + s);
    }
}

void sample_h_from_hprob(const rbm *m, const double *hprob, double *hsample){
    int i;
    double u;
    for(i = 0; i < m->nhidden; i++){
        u = random_double(0.0, 1.0); 
        if(u < hprob[i])
            hsample[i] = 1.0;
        else
            hsample[i] = 0.0;
    }
}

void get_vprob_given_hsample(const rbm *m, const double *hsample, double *vprob){
    int i, j;
    double s = 0;
    for(i = 0; i < m->nvisible; i++){
        s = 0;
        for(j = 0; j < m->nhidden; j++){
            s += m->W[j][i] * hsample[j]; 
        }
        vprob[i] = sigmoid(m->vbias[i] + s);
    }
}

void sample_v_from_vprob(const rbm *m, const double *vprob, double *vsample){
    int i;
    double u;
    for(i = 0; i < m->nvisible; i++){
        u = random_double(0.0, 1.0);
        if(u < vprob[i])
            vsample[i] = 1.0;
        else
            vsample[i] = 0.0;
    }
}

void gibbs_sampling_hvh(const rbm *m, const double *hsample, int step, double *end_hprob, double *end_hsample, double *end_vprob, double *end_vsample){
    int k;

    memcpy(end_hsample, hsample, m->nhidden * sizeof(double));

    //printf("origin:\n", k);
    //print_hsample(m, end_hsample);

    for(k = 0; k < step; k++){
        get_vprob_given_hsample(m, end_hsample, end_vprob);
        sample_v_from_vprob(m, end_vprob, end_vsample);
        get_hprob_given_vsample(m, end_vsample, end_hprob);
        sample_h_from_hprob(m, end_hprob, end_hsample);

        //printf("epcho %d:\n", k);
        //print_hsample(m, end_hsample);
    }
}

double get_free_energy(const rbm *m, const double *v){
    double vterm, hterm, s;
    int i, j; 
    for(i = 0, vterm = 0; i < m->nvisible; i++)
        vterm += v[i] * m->vbias[i];
    for(i = 0, hterm = 0; i < m->nhidden; i++){
        for(j = 0, s = 0.0; j < m->nvisible; j++){
            s += v[j] * m->W[i][j];
        }
        hterm += log(exp(m->hbias[i] + s) + 1.0);
    }
    return -vterm - hterm;
}

double get_pseudo_likelihood_cost(const rbm *m, dataset *d){
    int i, j;
    double old_val;
    double fe, fe_flip;
    double s;
    static int flip_idx = 0;

    for(i = 0, s = 0.0; i < d->N; i++){
        fe = get_free_energy(m, d->input[i]);

        old_val = d->input[i][flip_idx];
        d->input[i][flip_idx] = 1.0 - old_val;
        fe_flip = get_free_energy(m, d->input[i]);
        d->input[i][flip_idx] = old_val;

        s += m->nvisible * log(sigmoid(fe_flip - fe));
#ifdef DEBUG
        if((i+1) % 1000 == 0){
            printf("calculating pseudo likelihood cost! loop :%d\n", i);
        }
#endif
    }

    return s / d->N;
}

void print_model(const rbm *m, const dataset *d){
    printf("Weight:\n"); 
    int i, j;
    for(i = 0; i < m->nhidden; i++){
        printf("Hidden Node %d :\n", i);
        for(j = 0; j < m->nvisible; j++){
            printf("%lf%s", m->W[i][j], ((j+1) % d->ncol == 0 ? "\n" : "\t"));
        }
    }
    printf("VBias:\n");
    for(i = 0; i < m->nvisible; i++){
        printf("%lf%s", m->vbias[i], (i+1) % d->ncol == 0 ? "\n" : "\t");
    }
    printf("HBias:\n");
    for(i = 0; i < m->nhidden; i++){
        printf("%lf%s", m->hbias[i], (i+1) % 25 == 0 ? "\n" : "\t");
    }
}

void print_vsample(const rbm *m, const double *vsample){
    int i;
    for(i = 0; i < m->nvisible; i++){
        printf("%.2lf%s", vsample[i], (i+1) % 28 == 0 ? "\n" : "\t");
    }
}

void print_hsample(const rbm *m, const double *hsample){
    int i;
    for(i = 0; i < m->nhidden; i++){
        printf("%.2lf%s", hsample[i], (i+1) % 25 == 0 ? "\n" : "\t");
    }
}

int main(){
    int image_fd, label_fd;
    uint32_t magic_n, N, nrow, ncol, data;
    int nvisible, nhidden;
    int mini_batch, training_epcho;
    int i, j, k, p, q;
    int epcho;
    rbm m;
    dataset d;
    uint8_t x;
    double v1_sample[DEFAULT_MAXSIZE], v1_prob[DEFAULT_MAXSIZE], h1_sample[DEFAULT_MAXSIZE], h1_prob[DEFAULT_MAXSIZE];
    double v2_sample[DEFAULT_MAXSIZE], v2_prob[DEFAULT_MAXSIZE], h2_sample[DEFAULT_MAXSIZE], h2_prob[DEFAULT_MAXSIZE];
    double delta_W[DEFAULT_MAXSIZE][DEFAULT_MAXSIZE], delta_vbias[DEFAULT_MAXSIZE], delta_hbias[DEFAULT_MAXSIZE];
    double *chain_start = NULL;

    image_fd = open("../data/train-images-idx3-ubyte", O_RDONLY);
#ifndef DEBUG
    freopen("test.out", "w", stdout);
#endif

    if(image_fd == -1){
        fprintf(stderr, "cannot open file");
        exit(1);
    }

    /*
     * 按字节地读取文件
     while(1){
     read(image_fd, &x, sizeof(uint8_t));    
     printf("%u\n", x);
     }
     */

    read_uint32(image_fd, &magic_n);
    read_uint32(image_fd, &N);
    read_uint32(image_fd, &nrow);
    read_uint32(image_fd, &ncol);

#ifdef DEBUG
    printf("magic number: %u\nN: %u\nnrow: %u\nncol: %u\n", magic_n, N, nrow, ncol);
#endif

    init_dataset(&d, N, nrow, ncol);
    load_dataset_input(image_fd, &d);
    close(image_fd);

    srand(1234);
    nvisible = d.nrow * d.ncol;
    nhidden = 500;
    mini_batch = 20;
    training_epcho = 15;
    init_model(&m, nvisible, nhidden);

    print_vsample(&m, d.input[59999]);
    free_dataset(&d);
    free_model(&m);
    exit(0);
    
    for(epcho = 0; epcho < training_epcho; epcho++){

        for(k = 0; k < d.N / mini_batch; k++){
            for(j = 0; j < m.nhidden; j++){
                delta_hbias[j] = 0;
                for(p = 0; p < m.nvisible; p++){
                    delta_W[j][p] = 0; 
                }
            }
            for(p = 0; p < m.nvisible; p++){
                delta_vbias[p] = 0; 
            }
#ifdef DEBUG
            printf("epcho:%d\tstep:%d\n", epcho + 1, k + 1);
#endif

            for(i = 0; i < mini_batch; i++){

                memcpy(v1_sample, d.input[i], m.nhidden * sizeof(double));

                //printf("vsample:\n");
                //print_vsample(&m, v1_sample);

                get_hprob_given_vsample(&m, v1_sample, h1_prob);
                sample_h_from_hprob(&m, h1_prob, h1_sample);

                if(chain_start == NULL){
                    chain_start = h1_sample;
                }

                gibbs_sampling_hvh(&m, chain_start, 5, h2_prob, h2_sample, v2_prob, v2_sample);

                /**
                 * mini-batch调整delta
                 */
                for(j = 0; j < m.nhidden; j++){
                    delta_hbias[j] += h1_prob[j] - h2_prob[j];
                }
                for(j = 0; j < m.nvisible; j++){
                    delta_vbias[j] += v1_sample[j] - v2_sample[j];
                }
                for(j = 0; j < m.nhidden; j++){
                    for(p = 0; p < m.nvisible; p++){
                        delta_W[j][p] += h1_prob[j] * v1_sample[p] - h2_prob[j] * v2_sample[p];
                    }
                }

                chain_start = h2_sample;

                //printf("hprob:\n");
                //print_hsample(&m, h1_prob);
                //printf("hsample:\n");
                //print_hsample(&m, h1_sample);
            }

            /*
             * 根据delta调整参数
             */
            for(j = 0; j < m.nhidden; j++){
                m.hbias[j] += eta * delta_hbias[j] / mini_batch;
            }
            for(j = 0; j < m.nvisible; j++){
                m.vbias[j] += eta * delta_vbias[j] / mini_batch;
            }
            for(j = 0; j < m.nhidden; j++){
                for(p = 0; p < m.nvisible; p++){
                    m.W[j][p] += eta * delta_W[j][p] / mini_batch;
                }
            }

#ifdef DEBUG
            if((k+1) % 50 == 0){
                printf("cost: %.5lf\n", get_pseudo_likelihood_cost(&m, &d));
            }
#endif
        }
    }

    //print_dataset(&d);

    //print_model(&m, &d);

    free_dataset(&d);
    free_model(&m);
    return 0;
}
