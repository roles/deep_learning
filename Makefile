CC=gcc
CPP=g++
LDFLAGS=-lm -pg -lgfortran
CFLAGS=-c -g -DDEBUG -pg -Iinclude -I../include
OBJECTS=MLP.o RBM.o Logistic.o MLPLayer.o TrainModel.o MultiLayerTrainComponent.o TrainComponent.o Dataset.o Utility.o
OBJECTS:=$(patsubst %.o, src/%.o, $(OBJECTS))
MODELS=LogisticModel MLPModel RBMModel DBN
BLASLIB=./lib/libblas.a
CBLASLIB=./lib/libcblas.a
LOADER=gfortran
BITS=

$(MODELS) : % : src/%.o $(OBJECTS)
	$(CPP) $^ $(CBLASLIB) $(BLASLIB) $(LDFLAGS) -o $@

.cpp.o:
	${CPP} $(CFLAGS) $(LDFLAGS) -I include/ -o $@ $<

.c.o:
	${CC} $(CFLAGS) $(LDFLAGS) -I include/ -o $@ $<

.PHONY: clean blas test

blas:
ifeq ("$(BITS)", "32")
	cp -f ./lib/libblas32.a ./lib/libblas.a
	cp -f ./lib/libcblas32.a ./lib/libcblas.a
endif
ifeq ("$(BITS)", "64")
	cp -f ./lib/libblas64.a ./lib/libblas.a
	cp -f ./lib/libcblas64.a ./lib/libcblas.a
endif

clean:
	rm -rf $(MODELS) $(OBJECTS) *.out *.png *.txt

test:
	echo $(OBJECTS)
