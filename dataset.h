#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<stdint.h>
#include<arpa/inet.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define DEFAULT_MAXSIZE 1000

typedef struct dataset{
    uint32_t N;
    uint32_t nrow, ncol;
    double **input;
} dataset;

void init_dataset(dataset *d, uint32_t N, uint32_t nrow, uint32_t ncol);
void load_dataset_input(int fd, dataset *d);
void print_dataset(dataset *d);
void free_dataset(dataset *d);
void read_uint32(int fd, uint32_t *data);
