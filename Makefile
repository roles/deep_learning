CC=gcc
LDFLAGS=-lm
CFLAGS=-g -std=c99
OBJECTS=my_rbm.o

rbm: $(OBJECTS)
	${CC} -o rbm $(OBJECTS) $(LDFLAGS) $(CFLAGS)

clean:
	rm -rf *.o rbm *.out
