
CC      = gcc
CPP     = g++
CCFLAGS = -O3 -Wall 

MPICC = mpicxx
MPICCFLAGS = $(CCFLAGS)

all: mind-the-bind

help:
	@echo "make mind-the-bind"
	@echo "make all        mind-the-bind"
	@echo
	@echo "make clean"
	@echo "make cleanall"

mind-the-bind: mind-the-bind.c
	$(MPICC) $< -o $@.x $(MPICCFLAGS) -fopenmp

clean:
	rm -rf *.o *.log

cleanall: clean
	rm -rf *.x

