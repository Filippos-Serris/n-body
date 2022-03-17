#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define G 6.673e-11 // gravity constant
#define M 100       // body mass
#define Dt 0.03     // time between iterations
#define Exp 0.01    // softening parameter

__global__ void newPos(int *loop,int *N,double *C,double *V,double *F)
{
  int i,index,k;

  // dx/dy/dz: αποστάσεις ανα άξονα (x,y,z)
  // d: απόσταση δύο σωμάτων
  double dx,dy,dz,d;

  // dSquare: τετράγωνο απόστασης μεταξύ σωμάτων
  // mSquare: τετράγωνο μάζας σωμάτων
  // expSquare: τετράγωνο μεαβλητής απόσβεσης
  double dSquare,mSquare,expSquare;

  // a: επιτάχυνση σώματος
  double a;

  index=threadIdx.x+blockIdx.x*blockDim.x;

  mSquare=pow(M,2);
  expSquare=pow(Exp,2);

  // Υπολογισμοί
  for(i=0;i<*loop;i++)            // i: επαναλήψης πειράματως
  {
    if(index<*N)                  // index: σώμα στο οποίο ασκουντε δυνάμεις
    {
      for(k=0;k<*N;k++)           // k: σώμα το οποίο ασκεί δύναμη στο σώμα j
      {
        if(index!=k)
        {
          dx=C[k*3+0]-C[index*3+0];
          dy=C[k*3+1]-C[index*3+1];
          dz=C[k*3+2]-C[index*3+2];

          d=sqrt(pow(dx,2)+pow(dy,2)+pow(dz,2));
          dSquare=pow(d,2);

          F[index*3+0]-=G*mSquare*d/(pow(dSquare+expSquare,1.5)*dx);            // Αθροιστική δύναμη που ασκείται στο σώμα j
          F[index*3+1]-=G*mSquare*d/(pow(dSquare+expSquare,1.5)*dy);            // απο όλα τα υπόλοιπα σώματα k του συστήματος
          F[index*3+2]-=G*mSquare*d/(pow(dSquare+expSquare,1.5)*dz);            // στους τρίς άξονες (x,y,z)
        }
      }
    }

    if(index<*N)                                                // index: σώμα για το οποίο υπολογίζονται οι νέες συντεταγμένες
    {
      for(k=0;k<3;k++)                                          // k: συντεταγμένη που υπολογίζεται σε καθε κύκλο (x,y,z)
      {
        a=F[index*3+k]/M;                                       // Επιτάχυνση α=F/M

        F[index*3+k]=0.0;                                       // Επαναφορά της δύναμης F σε 0

        V[index*3+k]=V[index*3+k]+a*Dt;                         // Ταχύτητα V(n+1)=Vn+α*Δt
        C[index*3+k]=C[index*3+k]+V[index*3+k]*Dt;              // Θέση R(n+1)=Rn+Vn*Δt
      }
    }
  }
}

int main()
{
  // Μεταβλητές host
  int i;

  // h_N: πλήθος σωμάτων, h_loop: πλήθος επαναλήψεων
  int h_N,h_loop;

  // u/v: μεταβλητές ορισμού κανονικής κατανομής του x
  // x1: τιμή μεταξύ του [0,1] που ακολουθεί την κανονική κατανομή
  // r: μεταβλητή προσδιορισμού θέσης
  double u,v,x1,r;

  // x4/x5: τυχαίες μεταβλητές για τον ορισμό της vel
  // vesc: αρχική ταχύτητα (ταχύτητα αποφυγής)
  // vel: μεταβλητή προσδιορισμού ταχύτητας
  double x4,x5,vel,vesc;

  // theta: τυχαία τιμή μεταξύ [0,π], phi: τυχαία τιμή μεταξύ [0,2π]
  double theta,phi;

  // C: πίνακας συντεταγμένων, V: πίνακας ταχυτήτων, F: πίνακας δυνάμεων
  double *h_C,*h_V,*h_F;

  // Μεταβλητές device (αντίστοιχη λειτουργία όπως στον host)
  int *d_N,*d_loop;
  double *d_C,*d_F,*d_V;

  // start/stop: μεταβλητές μέτρησης χρόνου
  float time;
  cudaEvent_t start, stop;

  // Εισαγωγή δεδομένων απο τον χρήστη
  do
  {
    printf("\nInsert iterations: ");
    scanf("%d",&h_loop);
  }while(h_loop<10);

  do
  {
    printf("\nInsert body count: ");
    scanf("%d",&h_N);
  }while(h_N<2 || h_N>10);

  // Δέσμευση μνήμης για τον host
  h_C=(double*)malloc(h_N*3*sizeof(double));
  h_F=(double*)malloc(h_N*3*sizeof(double));
  h_V=(double*)malloc(h_N*3*sizeof(double));

  if(h_C==NULL || h_F==NULL || h_V==NULL)
  {
    printf("\nAllocation in CPU failed");
    return -1;
  }

  // Αρχικοπόιηση μευαβλητών
  for(i=0;i<h_N*3;i+=3)
  {
    // Ορισμός αρχικής θέσης
    do
    {
      u=(double)rand()/(double)RAND_MAX;
      v=(double)rand()/(double)RAND_MAX;

      x1=sqrt(8/M_E)*(v-0.5)/u;
    }while(pow(x1,2)>=-4*log(u) || x1<0 || x1>1);

    r=pow(pow(x1,-0.667)-1,-0.5);

    theta=(double)rand()/(double)RAND_MAX*M_PI;
    phi=(double)rand()/(double)RAND_MAX*2*M_PI;

    h_C[i]=r*sin(theta)*cos(phi);
    h_C[i+1]=r*sin(theta)*sin(phi);
    h_C[i+2]=r*cos(theta);

    // Ορισμός αρχικής ταχύτητας
    do
    {
      x4=rand();
      x5=rand();
    }while(0.1*x5>=pow(x4,2)*pow(1-pow(x4,2),3.5));

    vesc=sqrt(2)*pow(1+pow(r,2),-0.25);
    vel=vesc/x4;

    theta=(double)rand()/(double)RAND_MAX*M_PI;
    phi=(double)rand()/(double)RAND_MAX*2*M_PI;

    h_V[i]=vel*sin(theta)*cos(phi);
    h_V[i+1]=vel*sin(theta)*sin(phi);
    h_V[i+2]=vel*cos(theta);
  }

  for(i=0;i<h_N*3;i++)
  {
    h_F[i]=0.0;
  }

  // Αρχηκή θέση σωμάτων
  for(i=0;i<h_N*3;i+=3)
  {
    printf("\nMass %d\tC[x]: %.3f\tC[y]: %.3f\tC[z]: %.3f",i/3,h_C[i],h_C[i+1],h_C[i+2]);
  }

  printf("\n\n");

  // Έναρξη χρονομέτρησης
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Δέσμευση μνήμης για το device
  cudaMalloc((void**)&d_loop,sizeof(int));
  cudaMalloc((void**)&d_N,sizeof(int));

  cudaMalloc((void**)&d_C,h_N*3*sizeof(double));
  cudaMalloc((void**)&d_F,h_N*3*sizeof(double));
  cudaMalloc((void**)&d_V,h_N*3*sizeof(double));

  if(d_loop==NULL || d_N==NULL || d_C==NULL || d_F==NULL || d_V==NULL)
  {
    printf("\nAllocation in GPU failed");
    return -1;
  }

  // Αντιγραφή δεδομένων απο το host στο device
  cudaMemcpy(d_loop,&h_loop,sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_N,&h_N,sizeof(int),cudaMemcpyHostToDevice);

  cudaMemcpy(d_C,h_C,h_N*3*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_F,h_F,h_N*3*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_V,h_V,h_N*3*sizeof(double),cudaMemcpyHostToDevice);

  // Μεταβλητές παραλληλίας κάρτας γραφικών
  // threadsPerBlock: τα threads που θα εκτελεστουν σε κάθε block
  // blocksPerGrid: το πλήθος των blocks που περιέχονται μέσα σε ένα grid
  int threadsPerBlock = 256;
  int blocksPerGrid=(h_N+threadsPerBlock-1)/threadsPerBlock;

  newPos<<<blocksPerGrid,threadsPerBlock>>>(d_loop,d_N,d_C,d_V,d_F);

  // Αντιγραφή αποτελεσμάτων
  cudaMemcpy(h_C,d_C,h_N*3*sizeof(double),cudaMemcpyDeviceToHost);

  // Τέλος χρονομέτρησης
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);

  // Τελική θέση σωμάτων
  for(i=0;i<h_N*3;i+=3)
  {
    printf("\nMass %d\tC[x]: %.3f\tC[y]: %.3f\tC[z]: %.3f",i/3,h_C[i],h_C[i+1],h_C[i+2]);
  }
  printf("\n\nTime required is: %.3f\n",time*pow(10,-3));

  // Αποδέσμευση μνήμης host
  free(h_C);
  free(h_F);
  free(h_V);

  // Αποδέσμευση μνήμης device
  cudaFree(d_loop);
  cudaFree(d_N);
  cudaFree(d_C);
  cudaFree(d_F);
  cudaFree(d_V);

  return 0;
}
