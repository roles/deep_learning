CC=gcc
LDFLAGS=-lm
CFLAGS=-c -std=c99 -g -DDEBUG
OBJECTS=dataset.o rio.o ranlib.o rnglib.o
BLASLIB=./lib/libblas.a
CBLASLIB=./lib/libcblas.a
BLASLIB_FRANKLIN=./lib/blas_franklin.a
CBLASLIB_FRANKLIN=./lib/cblas_franklin.a
LOADER=gfortran

rbm: my_rbm.o $(OBJECTS)
	${CC} -o rbm my_rbm.o $(OBJECTS) -I ./include $(LDFLAGS)

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

rbm_blas_franklin: rbm_blas.o $(OBJECTS)
	$(LOADER) rbm_blas.o  $(OBJECTS) $(CBLASLIB_FRANKLIN) $(BLASLIB_FRANKLIN) -o $@
	
rsm_blas: rsm_blas.o $(OBJECTS)
	$(LOADER) rsm_blas.o  $(OBJECTS) $(CBLASLIB) $(BLASLIB) -o $@

classRBM_blas: classRBM_blas.o $(OBJECTS)
	$(LOADER) classRBM_blas.o  $(OBJECTS) $(CBLASLIB) $(BLASLIB) -o $@

classRBM_blas_franklin: classRBM_blas.o $(OBJECTS)
	$(LOADER) classRBM_blas.o  $(OBJECTS) $(CBLASLIB_FRANKLIN) $(BLASLIB_FRANKLIN) -o $@

test_cblas: test_cblas.o
	$(LOADER) test_cblas.o $(CBLASLIB) $(BLASLIB) -o $@

.c.o:
	${CC} $(CFLAGS) $(LDFLAGS) -I include/ -o $@ $<

clean:
	rm -rf *.o rbm msgd mlp test_cblas da_blas rbm_blas rsm_blas classRBM_blas *.out *.png *.txt
