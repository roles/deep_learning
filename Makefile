CC=gcc
LDFLAGS=-lm
CFLAGS=-c -O3 -std=c99 -g -DDEBUG
OBJECTS=dataset.o rio.o
BLASLIB=librefblas.a
LAPACKLIB=liblapack.a
LAPACKELIB=liblapacke.a
LOADER=gfortran

rbm: my_rbm.o $(OBJECTS)
	${CC} -o rbm my_rbm.o -I ./include $(OBJECTS) $(LDFLAGS) $(CFLAGS)

msgd: my_logistic_sgd.o $(OBJECTS)
	${CC} -o msgd my_logistic_sgd.o -I ./include $(OBJECTS) $(LDFLAGS) $(CFLAGS)

mlp: my_mlp.o my_logistic_sgd.o $(OBJECTS)
	${CC} -o mlp my_mlp.o my_logistic_sgd.o -I ./include $(OBJECTS) $(LDFLAGS) $(CFLAGS)

da: my_da.o $(OBJECTS)
	${CC} -o da my_da.o -I ./include $(OBJECTS) $(LDFLAGS) $(CFLAGS)

test_lapack: test_lapack.o
	$(LOADER) test_lapack.o lib/$(LAPACKELIB) lib/$(LAPACKLIB) lib/$(BLASLIB) -o $@

.c.o:
	${CC} $(CFLAGS) -I include/ -o $@ $<

clean:
	rm -rf *.o rbm msgd mlp *.out *.png *.txt
