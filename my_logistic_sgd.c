#include "dataset.h"

typedef struct log_reg {
    int n_in, n_out; 
    double ** W;
    double * b;
} log_reg;

void init_log_reg(log_reg *m, int n_in, int n_out){
        
}

int main(){
    int i, j, k;
    int train_x_fd, train_y_fd;
    uint32_t N, nrow, ncol, magic_n;
    dataset train_set;
    rio_t rio_train_x, rio_train_y;

    train_x_fd = open("../data/train-images-idx3-ubyte", O_RDONLY);
    train_y_fd = open("../data/train-labels-idx1-ubyte", O_RDONLY);

    if(train_x_fd == -1)
        fprintf(stderr, "cannot open train-images-idx3-ubyte\n");
    if(train_y_fd == -1)
        fprintf(stderr, "cannot open train-labels-idx1-ubyte\n");

    rio_readinitb(&rio_train_x, train_x_fd);
    rio_readinitb(&rio_train_y, train_y_fd);

    read_uint32(&rio_train_x, &magic_n);
    read_uint32(&rio_train_x, &N);
    read_uint32(&rio_train_x, &nrow);
    read_uint32(&rio_train_x, &ncol);
    init_dataset(&train_set, N, nrow, ncol);
    
    read_uint32(&rio_train_y, &magic_n);
    read_uint32(&rio_train_y, &N);
#ifdef DEBUG
    printf("magic number: %u\nN: %u\nnrow: %u\nncol: %u\n", magic_n, N, nrow, ncol);
    fflush(stdout);
#endif

    load_dataset_input(&rio_train_x, &train_set);
    load_dataset_output(&rio_train_y, &train_set);

    print_dataset(&train_set);

    close(train_x_fd);
    close(train_y_fd);
    
    free_dataset(&train_set);
    return 0;
}
