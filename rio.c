#include "rio.h"

void rio_readinitb(rio_t *rp, int fd, int type){        /* type表示是读buffer还是写buffer */
    rp->rio_fd = fd;
    if(type == 0){
        rp->rio_cnt = 0;
    }else{
        rp->rio_cnt = RIO_BUFFER_SIZE;
    }
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

static ssize_t rio_write(rio_t *rp, char *usrbuf, size_t n){
    int cnt;

    cnt = rp->rio_cnt;                  /* rp->rio_cnt表示未写入的字节数 */
    while(cnt <= 0){
        cnt = write(rp->rio_fd, rp->rio_buf, sizeof(rp->rio_buf));
        if(cnt < 0){
            if(errno != EINTR)
                return -1;
        }else{
            rp->rio_bufptr = rp->rio_buf;
            rp->rio_cnt = cnt;
        }
    }

    cnt = n;
    if(rp->rio_cnt < n)
        cnt = rp->rio_cnt;
    memcpy(rp->rio_bufptr, usrbuf, cnt);
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
    return (n - nleft);
}

ssize_t rio_writenb(rio_t *rp, void *usrbuf, size_t n){
    size_t nleft = n;
    ssize_t nwrite;
    char *bufp = usrbuf;

    while(nleft > 0){
        if((nwrite = rio_write(rp, bufp, nleft)) < 0){
            if(errno == EINTR)
                nwrite = 0;
            else
                return -1;
        }
        nleft -= nwrite;
        bufp += nwrite;
    }
    return (n - nleft);
}
