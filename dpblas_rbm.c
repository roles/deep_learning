#include"dataset.h"
#include"cblas.h"
#include<strings.h>
#include<time.h>

#define MAX_SIZE 17000
#define MAX_QUAR_SIZE 17000 * 9000
#define MAX_BATCH_SIZE 20
#define MAX_STEP 5000
#define eta 0.1
#define FOLD_NUM 10

#define TYPE_COUNT 18

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
double delta_W[MAX_QUAR_SIZE];
double delta_c[MAX_SIZE], delta_b[MAX_SIZE];
double t1[MAX_QUAR_SIZE], t2[MAX_QUAR_SIZE], Ivec[MAX_BATCH_SIZE * MAX_SIZE];

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

void dump_weight(FILE *weight_file, const rbm *m, int item_per_line, int hidden_unit_count){
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
    int item_per_line = 100;
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
    init_rbm(m, m->nvisible, m->nhidden);

    rio_readnb(&rbm_rio, m->W, m->nhidden * m->nvisible * sizeof(double));
    rio_readnb(&rbm_rio, m->b, m->nvisible * sizeof(double));
    rio_readnb(&rbm_rio, m->c, m->nhidden * sizeof(double));

    close(rbm_fd);
}

void train_rbm(rbm *m, const dataset_blas *train_set, const dataset_blas *validate_set, 
               const int mini_batch, const int n_epcho, const char *weight_filename){
    int i, j, k, epcho; 
    int batch_count, batch_size;
    int cd_k = 1;
    double *chain_start, *V1;
    double cost;
    FILE *weight_file;
    time_t start_time, end_time;

    weight_file = fopen(weight_filename, "w");
    if(weight_filename == NULL){
        fprintf(stderr, "no such file %s\n", weight_filename);
        exit(1);
    }
    batch_count = (train_set->N-1) / mini_batch + 1;
    chain_start = NULL;

    //dump_weight(weight_file, m);

    for(epcho = 0; epcho < n_epcho; epcho++){
        cost = 0;
        start_time = time(NULL);

        for(k = 0; k < batch_count; k++){
#ifdef DEBUG
            if((k+1) % 10 == 0){
                printf("epcho %d batch %d\n", epcho + 1, k + 1);
            }
#endif
            V1 = train_set->input + k * mini_batch * m->nvisible;

            if(k == (batch_count-1)){
                batch_size = train_set->N - mini_batch * k;
            }else{
                batch_size = mini_batch;
            }
            get_hprob(m, V1, Ph1, batch_size);
            get_hsample(m, Ph1, H1, batch_size);

            /*if(chain_start == NULL){
                chain_start = H1;
            }*/
            chain_start = H1;

            gibbs_sample_hvh(m, chain_start, H2, Ph2, V2, Pv2, cd_k, batch_size);

            //chain_start = H2;

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nhidden, m->nvisible, batch_size,
                        1.0, Ph2, m->nhidden, V2, m->nvisible,
                        0, t1, m->nvisible);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nhidden, m->nvisible, batch_size,
                        1.0, Ph1, m->nhidden, V1, m->nvisible,
                        -1, t1, m->nvisible);

            cblas_daxpy(m->nhidden * m->nvisible, eta / batch_size,
                        t1, 1, m->W, 1);

            cblas_dcopy(m->nvisible * batch_size, V1, 1, t1, 1);

            cblas_daxpy(m->nvisible * batch_size, -1, V2, 1, t1, 1);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nvisible, 1, batch_size,
                        eta / batch_size, t1, m->nvisible,
                        Ivec, 1, 1, m->b, 1);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nhidden, 1, batch_size,
                        1.0, Ph2, m->nhidden, Ivec, 1,
                        0, t1, 1);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nhidden, 1, batch_size,
                        1.0, Ph1, m->nhidden, Ivec, 1,
                        -1, t1, 1);

            cblas_daxpy(m->nhidden, eta / batch_size,
                        t1, 1, m->c, 1);

            /*cblas_dcopy(m->nhidden * batch_size, Ph1, 1, t1, 1);

            cblas_daxpy(m->nhidden * batch_size, -1, Ph2, 1, t1, 1);

            cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        m->nhidden, 1, batch_size,
                        eta / batch_size, t1, m->nvisible,
                        Ivec, 1, 1, m->c, 1);*/

            cost += get_PL(m, V1, batch_size);
        }

        end_time = time(NULL);
        printf("epcho %d cost: %.8lf\ttime: %.2lf min\n", epcho+1, cost / batch_count, (double)(end_time - start_time) / 60);

        //dump_weight(weight_file, m, m->nvisible, m->nhidden);
    }

    fclose(weight_file);
}

void test_rbm(char* folder_prefix, char* type){
    int i, j, k, p, q;
    int mini_batch = 10;
    int epcho, n_epcho = 15;
    int train_set_size = 50000, validate_set_size = 10000;
    dataset_blas train_set, validate_set; 
    rbm m;
    FILE *sample_file;
    char comb_dat_file[200], neg_dat_file[200];
    char comb_input_file[200], neg_input_file[200];
    char *dat_file, *input_file;

    int nhidden = 1000;

    sprintf(comb_dat_file, "%s%s/yc_comb.dat", folder_prefix, type);
    sprintf(neg_dat_file, "%s%s/yc_neg.dat", folder_prefix, type);
    sprintf(comb_input_file, "%s%s/yc_comb.txt", folder_prefix, type);
    sprintf(neg_input_file, "%s%s/yc_neg.txt", folder_prefix, type);

    for(i = 0; i < 2; i++){
        if(i == 0){
            dat_file = comb_dat_file;
            input_file = comb_input_file;
        }else{
            dat_file = neg_dat_file;
            input_file = neg_input_file;
        }
        srand(1234);

        /*
         * mnist dataset initialize
         */
        //load_mnist_dataset_blas(&train_set, &validate_set);
        //init_rbm(&m, 28*28, nhidden);

        load_tcga_dataset_blas(&train_set, input_file);
        init_rbm(&m, train_set.n_feature, nhidden);


        //load_rbm("rbm_model.dat", &m);

        train_rbm(&m, &train_set, &train_set, mini_batch, n_epcho, "tcga_rbm_weight.txt");
        dump_rbm(dat_file, &m);

        /*
         * dump sample
         *
        sample_file = fopen("rbm_sample.txt", "w");
        if(sample_file == NULL){
            fprintf(stderr, "cannot open rbm_sample.txt\n");
            exit(1);
        }
        dump_sample(sample_file, &m, validate_set.input + 100 * m.nvisible, 20);
        generate_sample(sample_file, &m, validate_set.input + 100 * m.nvisible, 20);

        fclose(sample_file);
        */

        free_rbm(&m);
        //free_dataset_blas(&validate_set);
        free_dataset_blas(&train_set);
    }
}

void cross_validation_train(){
    int i, j;    
    dataset_blas all_train_set, foldset[FOLD_NUM], train_set, *validate_set;
    rbm m;
    int mini_batch = 10;
    int n_epcho = 15, nhidden = 1000;

    srand(1234);
    for(i = 0; i < MAX_BATCH_SIZE * MAX_SIZE; i++)
        Ivec[i] = 1.0;

    load_tcga_dataset_blas(&all_train_set, "../data/yc/yc_table_comb.txt"); 
    init_rbm(&m, all_train_set.n_feature, nhidden);
    partition_trainset(&all_train_set, foldset, FOLD_NUM);

    train_set.input = (double*)malloc(all_train_set.N * all_train_set.n_feature * sizeof(double));

    combine_foldset(foldset, FOLD_NUM, 1, &train_set, validate_set);
    train_rbm(&m, &train_set, validate_set, mini_batch, n_epcho, "../data/yc_comb_weight.txt");
    
    dump_rbm("../data/yc/yc_model_cv1.dat", &m);

    free_dataset_blas(&all_train_set); 
    free(train_set.input);
    free_rbm(&m);
}

void cross_validation_reconstruct(){
    int i, j;    
    dataset_blas all_train_set, foldset[FOLD_NUM], train_set, *validate_set;
    rbm m;
    int mini_batch = 10;
    int n_epcho = 15, nhidden = 1000;

    load_tcga_dataset_blas(&all_train_set, "../data/yc_table_comb.txt"); 
    partition_trainset(&all_train_set, foldset, FOLD_NUM);

    train_set.input = (double*)malloc(all_train_set.N * all_train_set.n_feature * sizeof(double));

    combine_foldset(foldset, FOLD_NUM, 1, &train_set, validate_set);
    get_reconstruct_unit(validate_set, "../data/yc/yc_cv1_re.txt", "../data/yc/yc_model_cv1.dat");

    free_dataset_blas(&all_train_set); 
    free(train_set.input);
}

void get_hidden_unit(){
    rbm m;
    dataset_blas train_set;
    int i, j, k;
    FILE *hidden_unit_file;
    int batch_size, mini_batch = 20, batch_count;

    hidden_unit_file = fopen("hidden_unit.txt", "w+");
    if(hidden_unit_file == NULL){
        fprintf(stderr, "cannot open hidden_unit.txt\n");
        exit(1);
    }

    load_rbm("../data/tcga_rbm_model_7777.dat", &m);
    load_tcga_dataset_blas(&train_set, "../data/tcga_table.txt");

    batch_count = (train_set.N-1) / mini_batch + 1;
    for(i = 0; i < batch_count; i++){
#ifdef DEBUG
        printf("batch %d\n", i+1);
#endif
        if(i == batch_count-1){
            batch_size = train_set.N - i * mini_batch; 
        }else{
            batch_size = mini_batch;
        }
        get_hprob(&m, train_set.input + i * m.nvisible * mini_batch, Ph1, batch_size);
        for(j = 0; j < batch_size; j++){
            for(k = 0; k < m.nhidden; k++){
                fprintf(hidden_unit_file, "%.5lf%s", Ph1[j * m.nhidden + k],
                        k == m.nhidden-1 ? "\n" : "\t");
            }
        }
    }

    fclose(hidden_unit_file);
    free_rbm(&m);
    free_dataset_blas(&train_set);
}

void get_reconstruct_unit(dataset_blas *validate_set, char *re_filename, char *model_filename){
    rbm m;
    dataset_blas train_set;
    int i, j, k;
    FILE *re_file;
    int batch_size, mini_batch = 20, batch_count;

    re_file = fopen(re_filename, "w+");
    if(re_file == NULL){
        fprintf(stderr, "cannot open %s\n", re_filename);
        exit(1);
    }

    load_rbm(model_filename, &m);

    batch_count = (validate_set->N-1) / mini_batch + 1;
    for(i = 0; i < batch_count; i++){
        if(i == batch_count-1){
            batch_size = validate_set->N - i * mini_batch; 
        }else{
            batch_size = mini_batch;
        }
        gibbs_sample_vhv(&m, validate_set->input + i * m.nvisible * mini_batch,
                         H1, Ph1, V2, Pv2, 1, batch_size);
        for(j = 0; j < batch_size; j++){
            for(k = 0; k < m.nvisible; k++){
                fprintf(re_file, "%.5lf%s", Pv2[j * m.nvisible + k],
                        k == m.nvisible-1 ? "\n" : "\t");
            }
        }
    }

    fclose(re_file);
    free_rbm(&m);
}

void dump_all_weight(){
    rbm m;
    FILE *weight_file;

    load_rbm("../data/tcga_rbm_model_7777.dat", &m);
    weight_file = fopen("../data/tcga_weight.txt", "w+");

    dump_weight(weight_file, &m, 20947, 1000);
    
    fclose(weight_file);
    free_rbm(&m);
}

void test_reconstruct(char* folder_prefix, char* type){
    dataset_blas train_set;
    char comb_dat_file[200], neg_dat_file[200];
    char comb_output_file[200], neg_output_file[200];
    char comb_input_file[200], neg_input_file[200];

    sprintf(comb_dat_file, "%s%s/yc_comb.dat", folder_prefix, type);
    sprintf(neg_dat_file, "%s%s/yc_neg.dat", folder_prefix, type);
    sprintf(comb_input_file, "%s%s/yc_comb.txt", folder_prefix, type);
    sprintf(neg_input_file, "%s%s/yc_neg.txt", folder_prefix, type);
    sprintf(comb_output_file, "%s%s/yc_comb_re.txt", folder_prefix, type);
    sprintf(neg_output_file, "%s%s/yc_neg_re.txt", folder_prefix, type);
    
    printf("%s comb output\n", type);
    load_tcga_dataset_blas(&train_set, comb_input_file);
    get_reconstruct_unit(&train_set, comb_output_file, comb_dat_file);
    free_dataset_blas(&train_set);

    printf("%s neg output\n", type);
    load_tcga_dataset_blas(&train_set, neg_input_file);
    get_reconstruct_unit(&train_set, neg_output_file, neg_dat_file);
    free_dataset_blas(&train_set);
}


int main(){
    int i;
    char folder_prefix[] = "/home/wang/yys/data/yeast_cele/";
    //char folder_prefix[] = "/home/rolexye/project/Yeast_Cele/subprojects/";
    char *type_list[TYPE_COUNT] = {
        "yc_intermediate_enlarge_binary",
        "yc_intermediate_enlarge_identity",
        "yc_intermediate_enlarge_similarity",
        "yc_intermediate_integrate_binary",
        "yc_intermediate_integrate_identity",
        "yc_intermediate_integrate_similarity",
        "yc_intermediate_ortholog_binary",
        "yc_intermediate_ortholog_identity",
        "yc_intermediate_ortholog_similarity",
        "yc_stringent_enlarge_binary",
        "yc_stringent_enlarge_identity",
        "yc_stringent_enlarge_similarity",
        "yc_stringent_integrate_binary",
        "yc_stringent_integrate_identity",
        "yc_stringent_integrate_similarity",
        "yc_stringent_ortholog_binary",
        "yc_stringent_ortholog_identity",
        "yc_stringent_ortholog_similarity"

    };

    for(i = 0; i < MAX_BATCH_SIZE * MAX_SIZE; i++)
        Ivec[i] = 1.0;

    for(i = 0; i < TYPE_COUNT; i++){
        test_rbm(folder_prefix, type_list[i]);
        test_reconstruct(folder_prefix, type_list[i]);
        printf("%s completed\n", type_list[i]);
    }
    //test_reconstruct();
    //get_hidden_unit();
    //dump_all_weight();
    //cross_validation_train();
    //cross_validation_reconstruct();
}
