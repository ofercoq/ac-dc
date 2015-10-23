
#DISTRIBUTED_INCLUDE=-I.   -I /usr/include/gsl
#DISTRIBUTED_LIB_PATH=   -L /usr/lib


CC = g++  #change to use a different compiler
FC = gfortran

CXXFLAGS = -O3 -Wall -m64 -fopenmp
FFLAGS = -O -Wall -fbounds-check -g -Wno-uninitialized -m64

LIBS = -lgsl -lgslcblas -lgfortran

LBFGSBFOLDER = ~/src/Lbfgsb.3.0/

#============================================================================================
# You should not modify the lines below


cdn_prox:
	$(FC) -c $(LBFGSBFOLDER)linpack.f -o $(OBJFOLDER)linpack.o
	$(FC) -c $(LBFGSBFOLDER)blas.f -o $(OBJFOLDER)blas.o
	$(FC) -c $(LBFGSBFOLDER)timer.f -o $(OBJFOLDER)timer.o
	$(FC) -c $(LBFGSBFOLDER)lbfgsb.f -o $(OBJFOLDER)lbfgsb.o
	$(CC) -c $(CXXFLAGS) $(EXPFOLDER)cdn_prox.cpp -o $(OBJFOLDER)cdn_prox.o
	$(CC) $(CXXFLAGS) $(OBJFOLDER)linpack.o $(OBJFOLDER)blas.o $(OBJFOLDER)timer.o $(OBJFOLDER)lbfgsb.o $(OBJFOLDER)cdn_prox.o $(LIBS) -o $(BUILD_FOLDER)cdn_prox
#./$(BUILD_FOLDER)cdn_prox  -A ~/datasets/a1a  -R 0
