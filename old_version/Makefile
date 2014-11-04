CC=gcc
CPP=g++
LDFLAGS=-lm -pg -lgfortran
CFLAGS=-c -g -DDEBUG -pg -Iinclude -I../include
OBJECTS=RBM.o Logistic.o MLPLayer.o TrainModel.o Dataset.o Utility.o
BLASLIB=./lib/libblas.a
CBLASLIB=./lib/libcblas.a
BLASLIB_FRANKLIN=./lib/blas_franklin.a
CBLASLIB_FRANKLIN=./lib/cblas_franklin.a
BLASLIB_HELIX=./lib/blas_helix.a
CBLASLIB_HELIX=./lib/cblas_helix.a
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

rbm_blas_franklin: dpblas_rbm.o $(OBJECTS)
	$(LOADER) dpblas_rbm.o  $(OBJECTS) $(CBLASLIB_FRANKLIN) $(BLASLIB_FRANKLIN) -o $@

rbm_blas_helix: dpblas_rbm.o $(OBJECTS)
	$(LOADER) dpblas_rbm.o  $(OBJECTS) $(CBLASLIB_HELIX) $(BLASLIB_HELIX) -o $@
	
rsm_blas: rsm_blas.o $(OBJECTS)
	$(LOADER) rsm_blas.o  $(OBJECTS) $(CBLASLIB) $(BLASLIB) -o $@

classRBM_blas: classRBM_blas.o $(OBJECTS)
	$(LOADER) classRBM_blas.o  $(OBJECTS) $(CBLASLIB) $(BLASLIB) -o $@

classRBM_blas_franklin: classRBM_blas.o $(OBJECTS)
	$(LOADER) classRBM_blas.o  $(OBJECTS) $(CBLASLIB_FRANKLIN) $(BLASLIB_FRANKLIN) -o $@

test_cblas: test_cblas.o
	$(LOADER) test_cblas.o $(CBLASLIB) $(BLASLIB) -o $@

.cpp.o:
	${CPP} $(CFLAGS) $(LDFLAGS) -I include/ -o $@ $<

.c.o:
	${CC} $(CFLAGS) $(LDFLAGS) -I include/ -o $@ $<

clean:
	rm -rf *.o rbm msgd mlp test_cblas da_blas rbm_blas rsm_blas classRBM_blas DBN LogisticModel MLP *.out *.png *.txt
