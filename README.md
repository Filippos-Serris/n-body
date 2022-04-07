# N-body Problem in C Programming Language

## Introduction
N-Body problem is an astro-physics problem about the movement of the planets included in a solar system and their future position. In this thesis is
studied a very simple version of this problem. More spesific the interaction of 2 to 10 point masses with each other without taking in acount any collision 
or the existance of a solar sytem structure. The masses are randomly placed in a 3D cube. The code is written in C programming language and the purpose of
this thesis is to messure the execution time between different teknicks of the same implementation. The problem is being solved wirth brute-force method
and then by the same standards the problem implemets with the use of the Message Passing Interface(MPI), OpenMP, the combination of those two and at last
with the use of the Compute Unified Device Architecture (CUDA).
 

## Specifications
The messurments for the study took place at a single Computer so the results are not aqurate concerning the quantity of time but the percentance differnce 
between evere approach. The computer used for this simulation has the following spesifications:
- CPU: AMD Ryzen 2400g
- RAM: 8gb DDR4, 3000MHz
- GPU: Nvidia GT1060Ti
- Operatin System: 

The system specifications about the necessarily tools for the same execution are referesed below
- MPI: 4.0.2
- OpenMP: 4.5
- CUDA: CUDA SDK 10.02

## Instalation Guide

### MPI
- Download the the needed files from [here](https://www.mpich.org/downloads/).
- Unzip the file with the following command " tar -xzf mpich2_version.tar.gz "
- Go inside the directore that you unziped the files " cd file-name "
- Execute the following commands 
  1. ./configure --disable -fortran "
  2. sudo make install
  3. mpiexec --version (Expecting outcome: Version: 4.0.2 for example)

At this point MPI is intalled and redy for use.

### OpenMP
OpenMP is propably installed in your system try this command " echo |cpp -fopenmp -dM |grep -i open " the expectin output is something like that 
" #define _OPENMP 201511 " by googling the number can identify the OpenMP version of yours.

### CUDA
Cuda instalation is different accordingly the GPU that is insalled in the current sytem so a complite guide is placed [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 


## Execution Guide

### Execution of basic program (brute-force C)
-gcc -c serial.c
-gcc -o serila serial -lm
-./serial

### Execution MPI
1. mpicc mpi.c -o mpi -lm
2. mpiexec -npX ./mpi

### OpenMP
1. gcc open.c -o open -fopenmp -lm
2. ./open

### Combination of MPI-OpenMP
1. mpicc mpi-open.c -o mpi-open -fopenmp -lm
2. mpirun -npX ./mpi-open

### CUDA
1. nvcc -arch sm_75 cuda.cu -o cuda
2. ./cuda
