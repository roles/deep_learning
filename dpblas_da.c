#include"dataset.h"
#include"cblas.h"

void test_da(){
    dataset_blas train_set, validate_set; 

    load_mnist_dataset_blas(&train_set, &validate_set);

    print_dataset_blas(&train_set);

    free_dataset_blas(&train_set);
    free_dataset_blas(&validate_set);
}

int main(){
    test_da();
    return 0;
}
