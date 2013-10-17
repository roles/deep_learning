#include<unistd.h>
#include<errno.h>

#ifndef RIO_H

#define RIO_H
#define RIO_BUFFER_SIZE 8192

typedef struct {
    int rio_fd;
    int rio_cnt;            /* unread bytes in internal buffer */
    char *rio_bufptr;       /* next unread byte in internal buffer */
    char rio_buf[RIO_BUFFER_SIZE];
} rio_t;

void rio_readinitb(rio_t *rp, int fd);
ssize_t rio_readnb(rio_t *rp, void *usrbuf, size_t n);

#endif
