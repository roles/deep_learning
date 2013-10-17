CC=gcc
LDFLAGS=-lm
CFLAGS=-g -std=c99 -DDEBUG
OBJECTS=my_rbm.o dataset.o

rbm: $(OBJECTS)
	${CC} -o rbm $(OBJECTS) $(LDFLAGS) $(CFLAGS)

clean:
	rm -rf *.o rbm *.out *.png *.txt
