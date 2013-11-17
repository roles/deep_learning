CC=gcc
LDFLAGS=-lm
CFLAGS=-c -std=c99 -g -DDEBUG
OBJECTS=dataset.o rio.o
BLASLIB=./lib/libblas.a
CBLASLIB=./lib/libcblas.a
LOADER=gfortran

rbm: my_rbm.o $(OBJECTS)
	${CC} -o rbm my_rbm.o -I ./include $(OBJECTS) $(LDFLAGS) $(CFLAGS)

msgd: my_logistic_sgd.o $(OBJECTS)
	${CC} -o msgd my_logistic_sgd.o -I ./include $(OBJECTS) $(LDFLAGS) $(CFLAGS)

mlp: my_mlp.o my_logistic_sgd.o $(OBJECTS)
	${CC} -o mlp my_mlp.o my_logistic_sgd.o -I ./include $(OBJECTS) $(LDFLAGS) $(CFLAGS)

da: my_da.o $(OBJECTS)
	${CC} -o da my_da.o -I ./include $(OBJECTS) $(LDFLAGS)

da_blas: dpblas_da.o $(OBJECTS)
	$(LOADER) dpblas_da.o $(OBJECTS) $(CBLASLIB) $(BLASLIB) -o $@

rbm_blas: dpblas_rbm.o $(OBJECTS)
	$(LOADER) dpblas_rbm.o $(OBJECTS) $(CBLASLIB) $(BLASLIB) -o $@

test_cblas: test_cblas.o
	$(LOADER) test_cblas.o $(CBLASLIB) $(BLASLIB) -o $@

.c.o:
	${CC} $(CFLAGS) -I include/ -o $@ $<

clean:
	rm -rf *.o rbm msgd mlp test_cblas da_blas rbm_blas *.out *.png *.txt
