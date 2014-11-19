CC=gcc
CPP=g++
BITS=32
MKLLDFLAGS_32=-L/opt/intel/composer_xe_2013.5.192/mkl/lib/ia32 /opt/intel/composer_xe_2013.5.192/mkl/lib/ia32/libmkl_intel.a \
		   -Wl,--start-group \
		   /opt/intel/composer_xe_2013.5.192/mkl/lib/ia32/libmkl_intel_thread.a \
		   /opt/intel/composer_xe_2013.5.192/mkl/lib/ia32/libmkl_core.a \
		   -Wl,--end-group \
		   -L/opt/intel/composer_xe_2013.5.192/compiler/lib/ia32 \
		   -liomp5 -lpthread -ldl -lm
MKLLDFLAGS_64=-L/opt/intel/composer_xe_2013.5.192/mkl/lib/intel64 /opt/intel/composer_xe_2013.5.192/mkl/lib/intel64/libmkl_intel_lp64.a \
		   -Wl,--start-group \
		   /opt/intel/composer_xe_2013.5.192/mkl/lib/intel64/libmkl_intel_thread.a \
		   /opt/intel/composer_xe_2013.5.192/mkl/lib/intel64/libmkl_core.a \
		   -Wl,--end-group \
		   -L/opt/intel/composer_xe_2013.5.192/compiler/lib/intel64 \
		   -liomp5 -lpthread -ldl -lm
LDFLAGS=-lm -pg -lgfortran

ifeq ("$(BITS)", "32")
LDFLAGS:=$(LDFLAGS) $(MKLLDFLAGS_32)
endif
ifeq ("$(BITS)", "64")
LDFLAGS:=$(LDFLAGS) $(MKLLDFLAGS_64)
endif

CFLAGS=-c -g -DDEBUG -pg -Iinclude -I../include -I/opt/intel/composer_xe_2013.5.192/mkl/include
OBJECTS=MultiLayerRBM.o MLP.o RBM.o Logistic.o MLPLayer.o TrainModel.o MultiLayerTrainComponent.o TrainComponent.o Dataset.o Utility.o
OBJECTS:=$(patsubst %.o, src/%.o, $(OBJECTS))
MKLOBJECTS=Logistic.o MLPLayer.o RBM.o
MKLOBJECTS:=$(patsubst %, src/%, $(MKLOBJECTS))
MODELS=LogisticModel MLPModel RBMModel DBN
BLASLIB=./lib/libblas.a
CBLASLIB=./lib/libcblas.a
LOADER=gfortran

#$(CPP) $^ $(CBLASLIB) $(BLASLIB) $(LDFLAGS) -o $@

$(MODELS) : % : src/%.o $(OBJECTS)
	$(CPP) $^ $(LDFLAGS) -o $@

$(MKLOBJECTS) : %.o : %.cpp
	/opt/intel/bin/icpc $(CFLAGS) -I include -o $@ $<

.cpp.o:
	${CPP} $(CFLAGS) -I include/ -o $@ $<

.c.o:
	${CC} $(CFLAGS) -I include/ -o $@ $<

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
	rm -rf $(MODELS) $(OBJECTS) *.out *.png *.txt src/*.o

test:
	echo $(MKLOBJECTS)
