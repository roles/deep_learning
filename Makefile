CC=gcc
CPP=g++
LDFLAGS=-lm -pg -lgfortran
CFLAGS=-c -g -DDEBUG -pg -Iinclude -I../include
OBJECTS=RBM.o Logistic.o MLPLayer.o TrainModel.o Dataset.o Utility.o
BLASLIB=./lib/libblas.a
CBLASLIB=./lib/libcblas.a
LOADER=gfortran
BITS=


DBN: DBN.o $(OBJECTS)
	$(CPP) DBN.o $(OBJECTS) $(CBLASLIB) $(BLASLIB) $(LDFLAGS) -o $@

LogisticModel: LogisticModel.o $(OBJECTS)
	$(CPP) LogisticModel.o $(OBJECTS) $(CBLASLIB) $(BLASLIB) $(LDFLAGS) -o $@

MLP: MLP.o $(OBJECTS)
	$(CPP) MLP.o $(OBJECTS) $(CBLASLIB) $(BLASLIB) $(LDFLAGS) -o $@

.cpp.o:
	${CPP} $(CFLAGS) $(LDFLAGS) -I include/ -o $@ $<

.c.o:
	${CC} $(CFLAGS) $(LDFLAGS) -I include/ -o $@ $<

.PHONY: clean blas

	#	cp -f ./lib/libblas32.a ./lib/libblas.a
	#	cp -f ./lib/libcblas32.a ./lib/libcblas.a
	#else
	#	ifeq ($(BITS),64)
	#	cp -f ./lib/libblas64.a ./lib/libblas.a
	#	cp -f ./lib/libcblas64.a ./lib/libcblas.a
	#	else
	#	endif
	#endif
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
	rm -rf *.o DBN LogisticModel MLP *.out *.png *.txt

