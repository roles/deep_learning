#include "rio.h"

void rio_readinitb(rio_t *rp, int fd){
    rp->rio_fd = fd;
    rp->rio_cnt = 0;
    rp->rio_bufptr = rp->rio_buf;
}

static ssize_t rio_read(rio_t *rp, char *usrbuf, size_t n){
    int cnt;

    while(rp->rio_cnt <= 0){
        rp->rio_cnt = read(rp->rio_fd, rp->rio_buf, sizeof(rp->rio_buf));

        if(rp->rio_cnt < 0){
            if(errno != EINTR)
                return -1;
        }
        else if(rp->rio_cnt == 0)       /* EOF */
            return 0;
        else
            rp->rio_bufptr = rp->rio_buf;
    }

    /* copy min(n, rp->rio_cnt) bytes from interal buf to user buf */
    cnt = n;
    if(rp->rio_cnt < n)
        cnt = rp->rio_cnt;
    memcpy(usrbuf, rp->rio_bufptr, cnt);
    rp->rio_bufptr += cnt;
    rp->rio_cnt -= cnt;
    return cnt;
}

ssize_t rio_readnb(rio_t *rp, void *usrbuf, size_t n){
    size_t nleft = n;
    ssize_t nread;
    char *bufp = usrbuf;
    
    while(nleft > 0){
        if((nread = rio_read(rp, bufp, nleft)) < 0){
            if(errno == EINTR)
                nread = 0;      /* interrupted by sig handler return, call read() again */
            else
                return -1;
        }
        else if(nread == 0)
            break;              /* EOF */
        nleft -= nread;
        bufp += nread;
    }
}
