#include"dataset.h"
#include"my_logistic_sgd.h"

#define MAX_HIDDEN_SIZE 1000
#define MAX_LAYER 10
#define eta 0.01
#define L2_reg 0.0001

typedef struct hlayer{
    int n_in, n_out;
    double **W;
    double *b;
} hlayer;

double grad_b[MAX_HIDDEN_SIZE];
double **grad_w;
double delta_w[MAX_LAYER][MAX_HIDDEN_SIZE][MAX_HIDDEN_SIZE], delta_b[MAX_LAYER][MAX_HIDDEN_SIZE];
double prob_y[MAX_HIDDEN_SIZE];
double target[MAX_HIDDEN_SIZE];
double **input;
double **output;
double local_output[MAX_LAYER][MAX_HIDDEN_SIZE];
double d[MAX_LAYER][MAX_HIDDEN_SIZE];

void init_hlayer(hlayer *hl, int n_in, int n_out){
    int i, j;
    double low, high;

    low = -4 * sqrt((double)6 / (n_in + n_out));
    high = 4 * sqrt((double)6 / (n_in + n_out));

    hl->n_in = n_in;
    hl->n_out = n_out;
    hl->W = (double**)malloc(n_out * sizeof(double*));
    for(i = 0; i < n_out; i++){
        hl->W[i] = (double*)malloc(n_in * sizeof(double)); 
    }
    for(i = 0; i < n_out; i++){
        for(j = 0; j < n_in; j++){
            hl->W[i][j] = random_double(low, high);
        }
    }
    hl->b = (double*)calloc(n_out, sizeof(double));
}

void free_hlayer(hlayer *hl){
    int i;
    for(i = 0; i < hl->n_out; i++)
        free(hl->W[i]);
    free(hl->W);
    free(hl->b);
}

void get_sigmoid_y_given_x(const hlayer *hl, const double *x, double *y){
    int i, j; 
    double a, s;

    for(i = 0; i < hl->n_out; i++){
        for(j = 0, a = 0.0; j < hl->n_in; j++){
            a += x[j] * hl->W[i][j];
        }
        y[i] = sigmoid(a);
    }
}

void get_tanh_y_given_x(const hlayer *hl, const double *x, double *y){
    int i, j; 
    double a, s;

    for(i = 0; i < hl->n_out; i++){
        for(j = 0, a = 0.0; j < hl->n_in; j++){
            a += x[j] * hl->W[i][j];
        }
        y[i] = tanh(a);
    }
}


void get_mlp_delta(const hlayer *hl, const double *y, const double *d_high, double *d_low, int size){
    int i, j;
    double derivative, s;

    for(i = 0; i < size; i++){
        derivative = get_sigmoid_derivative(y[i]);
        for(j = 0, s = 0.0; j < hl->n_out; j++){
            s += d_high[j] * hl->W[j][i];
        }
        d_low[i] = derivative * s;
    }
}

void get_mlp_loss_error(const hlayer **hl, int hidden_layer, const double **x, const uint8_t *t, const int size, double *loss, int *error){
    int i, j, k, l;
    int pred_y;
    double s;
    double max_y;
    double **input, **output;
    double *y;

    *loss = 0.0;
    *error = 0;
    bzero(target, sizeof(target));
    input = (double**)malloc((hidden_layer + 1) * sizeof(double*));
    output = (double**)malloc((hidden_layer + 1) * sizeof(double*));

    /* feed-forward */

    for(i = 0; i < size; i++){
        for(l = 0; l < hidden_layer + 1; l++){
            if(l == 0){
                input[l] = x[i];
            }else{
                input[l] = output[l-1];
            }
            output[l] = local_output[l];

            if(l == hidden_layer){
                get_softmax_y_given_x((log_reg*)hl[l], input[l], output[l]);
            }else{
                get_sigmoid_y_given_x(hl[l], input[l], output[l]);
            }
        }
        y = output[hidden_layer];
        for(j = 0, max_y = -1.0; j < hl[hidden_layer]->n_out; j++){
            if(y[j] > max_y){
                max_y = y[j];
                pred_y = j;
            }
        }
        target[t[i]] = 1.0;
        for(j = 0, s = 0.0; j < hl[hidden_layer]->n_out; j++){
            s += target[j] * log(y[j]); 
        }
        target[t[i]] = 0.0;
        *loss += s;
        *error += (pred_y == t[i] ? 0 : 1);
    }
    *loss = -(*loss) / size;

    free(input);
    free(output);
}

void train_mlp(){
    int i, j, k, p, q, l;
    int train_x_fd, train_y_fd;
    int train_set_size = 50000, validate_set_size = 10000;
    int mini_batch = 20;
    int epcho, n_epcho = 1000;
    int hidden_size[] = {500};
    int layer_size[MAX_LAYER];
    int hidden_layer = sizeof(hidden_size) / sizeof(int);
    hlayer layers[MAX_LAYER];
    hlayer **mlp_layers;
    int n_in, n_out;
    int n_labels = 10;


    uint32_t N, nrow, ncol, magic_n;
    dataset train_set, validate_set;
    rio_t rio_train_x, rio_train_y;
    log_reg m;

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
    srand(1234);

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

    /* 初始化每一个hidden layer */
    mlp_layers = (hlayer**)malloc((hidden_layer + 1) * sizeof(hlayer*));

    for(i = 0; i < hidden_layer + 1; i++){
        mlp_layers[i] = &layers[i]; 

        if(i == 0){
            n_in = nrow * ncol;
            layer_size[i] = nrow * ncol;
        }else{
            n_in = hidden_size[i-1];
            layer_size[i] = hidden_size[i-1];
        }

        if(i == hidden_layer){
            n_out = n_labels;
            init_log_reg((log_reg*)mlp_layers[i], n_in, n_out);
        }else{
            n_out = hidden_size[i];
            init_hlayer(mlp_layers[i], n_in, n_out);
        }
    }
    layer_size[hidden_layer+1] = n_labels;

    /* input, output作为迭代时的中间量 */
    input = (double**)malloc((hidden_layer + 1) * sizeof(double*));
    output = (double**)malloc((hidden_layer + 1) * sizeof(double*));
    bzero(target, sizeof(target));

#ifdef DEBUG
    get_mlp_loss_error(mlp_layers, hidden_layer, validate_set.input, validate_set.output, validate_set_size, &loss, &error);
    printf("origin loss :%.5lf\t error :%d\n", loss, error);
#endif

    for(epcho = 0; epcho < n_epcho; epcho++){

        for(k = 0; k < train_set.N / mini_batch; k++){
            bzero(delta_w, sizeof(delta_w));
            bzero(delta_b, sizeof(delta_b));

            for(i = 0; i < mini_batch; i++){

                target[train_set.output[i]] = 1.0;
                /* feed-forward */
                for(l = 0; l < hidden_layer + 1; l++){
                    if(l == 0){
                        input[l] = train_set.input[i];
                    }else{
                        input[l] = output[l-1];
                    }
                    output[l] = local_output[l];

                    if(l == hidden_layer){
                        get_softmax_y_given_x((log_reg*)mlp_layers[l], input[l], output[l]);
                    }else{
                        get_sigmoid_y_given_x(mlp_layers[l], input[l], output[l]);
                    }

                }

                /* back-propagation */
                for(l = hidden_layer; l > 0; l--){
                    if(l == hidden_layer){
                        get_log_reg_delta(output[l], target, d[l], mlp_layers[l]->n_out);
                    }else{
                        get_mlp_delta(mlp_layers[l+1], output[l], d[l+1], d[l], mlp_layers[l]->n_out);
                    }

                    for(j = 0; j < mlp_layers[l]->n_out; j++){
                        for(p = 0; p < mlp_layers[l]->n_in; p++){
                            delta_w[l][j][p] += d[l][j] * input[l][p] + 2 * L2_reg * mlp_layers[l]->W[j][p];
                        }
                        delta_b[l][j] += d[l][j] + 2 * L2_reg * mlp_layers[l]->b[j];
                    }
                }
                target[train_set.output[i]] = 0.0;
            }

            /* modify parameter */
            for(l = 0; l < hidden_layer + 1; l++){ 
                for(j = 0; j < mlp_layers[l]->n_out; j++){
                    for(p = 0; p < mlp_layers[l]->n_in; p++){
                        mlp_layers[l]->W[j][p] -= eta * delta_w[l][j][p] / mini_batch;
                    }
                    mlp_layers[l]->b[j] -= eta * delta_b[l][j] / mini_batch;
                }
            }
#ifdef DEBUG
            if((k+1) % 500 == 0){
                printf("epcho %d batch %d\n", epcho + 1, k + 1);
            }
#endif
        }
#ifdef DEBUG
        get_mlp_loss_error(mlp_layers, hidden_layer, validate_set.input, validate_set.output, validate_set_size, &loss, &error);
        printf("epcho %d loss :%.5lf\t error :%d\n", epcho + 1, loss, error);
#endif
    }

    close(param_fd);
    close(train_x_fd);
    close(train_y_fd);

    free(input);
    free(output);
    free_dataset(&train_set);
    free_dataset(&validate_set);
    for(i = 0; i < hidden_layer + 1; i++)
        free_hlayer(mlp_layers[i]);
    free(mlp_layers);
}

int main(){
    train_mlp();

    return 0;
}
