CC=gcc
LDFLAGS=-lm
CFLAGS=-g -std=c99 -DDEBUG
OBJECTS=dataset.o rio.o

rbm: my_rbm.o $(OBJECTS)
	${CC} -o rbm my_rbm.o $(OBJECTS) $(LDFLAGS) $(CFLAGS)

msgd: my_logistic_sgd.o $(OBJECTS)
	${CC} -o msgd my_logistic_sgd.o $(OBJECTS) $(LDFLAGS) $(CFLAGS)

mlp: my_mlp.o my_logistic_sgd.o $(OBJECTS)
	${CC} -o mlp my_mlp.o my_logistic_sgd.o $(OBJECTS) $(LDFLAGS) $(CFLAGS)

da: my_da.o $(OBJECTS)
	${CC} -o da my_da.o $(OBJECTS) $(LDFLAGS) $(CFLAGS)

clean:
	rm -rf *.o rbm msgd mlp *.out *.png *.txt
