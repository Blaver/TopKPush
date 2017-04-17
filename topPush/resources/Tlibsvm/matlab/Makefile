# This Makefile is used under Linux

MATLABDIR ?= /usr/local/matlab
# for Mac
# MATLABDIR ?= /opt/local/matlab

CXX ?= g++
#CXX = g++-4.1
CFLAGS = -Wall -Wconversion -O3 -fPIC -I$(MATLABDIR)/extern/include -I..

MEX = $(MATLABDIR)/bin/mex
MEX_OPTION = CC="$(CXX)" CXX="$(CXX)" CFLAGS="$(CFLAGS)" CXXFLAGS="$(CFLAGS)"
# comment the following line if you use MATLAB on 32-bit computer
MEX_OPTION += -largeArrayDims
MEX_EXT = $(shell $(MATLABDIR)/bin/mexext)

all:	matlab

matlab:	binary

octave:
	@echo "please type make under Octave"

binary: svmpredict.$(MEX_EXT) svmtrain.$(MEX_EXT) libsvmread.$(MEX_EXT) libsvmwrite.$(MEX_EXT)

svmpredict.$(MEX_EXT):     svmpredict.cpp ../svm.h ../svm.o svm_model_matlab.o
	$(MEX) $(MEX_OPTION) svmpredict.cpp ../svm.o svm_model_matlab.o

svmtrain.$(MEX_EXT):       svmtrain.cpp ../svm.h ../svm.o svm_model_matlab.o
	$(MEX) $(MEX_OPTION) svmtrain.cpp ../svm.o svm_model_matlab.o

libsvmread.$(MEX_EXT):	libsvmread.cpp
	$(MEX) $(MEX_OPTION) libsvmread.cpp

libsvmwrite.$(MEX_EXT):	libsvmwrite.cpp
	$(MEX) $(MEX_OPTION) libsvmwrite.cpp

svm_model_matlab.o:     svm_model_matlab.cpp ../svm.h
	$(CXX) $(CFLAGS) -c svm_model_matlab.cpp

../svm.o: ../svm.cpp ../svm.h
	make -C .. svm.o

clean:
	rm -f *~ *.o *.mex* *.obj ../svm.o
