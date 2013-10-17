CC=gcc
LDFLAGS=-lm
CFLAGS=-g -std=c99 -DDEBUG
OBJECTS=dataset.o rio.o

rbm: my_rbm.o $(OBJECTS)
	${CC} -o rbm my_rbm.o $(OBJECTS) $(LDFLAGS) $(CFLAGS)

msgd: my_logistic_sgd.o $(OBJECTS)
	${CC} -o msgd my_logistic_sgd.o $(OBJECTS) $(LDFLAGS) $(CFLAGS)

clean:
	rm -rf *.o rbm *.out *.png *.txt
