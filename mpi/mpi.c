#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define G 6.673e-11 // gravity constant
#define M 100       // body mass
#define Dt 0.03     // time between iterations
#define Exp 0.01    // softening parameter

int main(int argc, char **argv)
{
  int i, j, k;

  // N: bodies multitude, loop: iterations multitude
  int N, loop;

  // u/v: variables defining if x1 follows normal distribution
  // x1: variable between [0,1] that follows normal distribution
  // r: variable responsible for defining body starting position
  double u, v, x1, r;

  // x4/x5: random values for defining vel value
  // vesc: escape velocity
  // vel: variable responsible for defining body starting velocity
  double x4, x5, vel, vesc;

  // theta: random value between [0,π], phi: random value between [0,2π]
  double theta, phi;

  // C: coordinents table, Cw: coordinents table for every process
  // V: velocity table, Vw: velocity table for every process
  // Fw: force table for every process
  double *C, *Cw, *V, *Vw, *Fw;

  // dx/dy/dz: distance in axis (x,y,z), d: distance of two bodies
  double dx, dy, dz, d;

  // dSquare: square of d
  // mSquare: square of body mass
  // expSquare: square of softening parameter
  double dSquare, mSquare, expSquare;

  // a: body acceleration
  double a;

  // ts/te: time mesurement variables
  double ts, te;

  // rank: process id
  // size: set of processes
  int rank, size;

  // part: integer division of data
  // remain: division remainder
  int part, remain;

  // start: process start point of data
  // end: process end point of data
  // multitude: multitude of bodies a process is responsible for
  // length: multitude of data a process is responsible for
  int start, end, multitude, length;

  // Counts: table of amount of data for every process
  // StartPoint: start point for every process in the original tables of process 0
  // EndPoint: start point for every process in the original tables of process 0
  int *Counts, *StartPoint, *EndPoint;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // gather input data from the user
  if (rank == 0)
  {
    do
    {
      printf("\nInsert iterations: ");
      scanf("%d", &loop);
    } while (loop < 10);

    do
    {
      printf("\nInsert body count: ");
      scanf("%d", &N);
    } while (N < 2 || N > 10);

    part = N / size;
    remain = N % size;
  }

  // timing starts
  ts = MPI_Wtime();

  // transfering data from master to all processes
  MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&loop, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(&part, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&remain, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // body range for every process
  if (rank < remain)
  {
    start = rank * (part + 1);
    end = start + part;
  }
  else
  {
    start = rank * part + remain;
    end = start + (part - 1);
  }

  multitude = end - start + 1;
  length = multitude * 3;

  // memory allocation
  C = malloc(N * 3 * sizeof(double));
  Cw = malloc(length * sizeof(double));

  if (rank == 0)
  {
    V = malloc(N * 3 * sizeof(double));
  }

  Vw = malloc(length * sizeof(double));
  Fw = malloc(length * sizeof(double));

  Counts = malloc(size * sizeof(int));
  StartPoint = malloc(size * sizeof(int));
  EndPoint = malloc(size * sizeof(int));

  if (rank == 0 && V == NULL)
  {
    printf("\nAllocation failed\nSimulation aborted");
    return -1;
  }

  if (C == NULL || Cw == NULL || Vw == NULL || Fw == NULL || Counts == NULL || StartPoint == NULL || EndPoint == NULL)
  {
    printf("\nAllocation failed\nSimulation aborted");
    return -1;
  }

  // input data initialization
  if (rank == 0)
  {
    for (i = 0; i < N * 3; i += 3)
    {
      // calculating starting position
      do
      {
        u = (double)rand() / (double)RAND_MAX;
        v = (double)rand() / (double)RAND_MAX;

        x1 = sqrt(8 / M_E) * (v - 0.5) / u;
      } while (pow(x1, 2) >= -4 * log(u) || x1 < 0 || x1 > 1);

      r = pow(pow(x1, -0.667) - 1, -0.5);

      theta = (double)rand() / (double)RAND_MAX * M_PI;
      phi = (double)rand() / (double)RAND_MAX * 2 * M_PI;

      C[i] = r * sin(theta) * cos(phi);
      C[i + 1] = r * sin(theta) * sin(phi);
      C[i + 2] = r * cos(theta);

      // calculating starting velocity
      do
      {
        x4 = rand();
        x5 = rand();
      } while (0.1 * x5 >= pow(x4, 2) * pow(1 - pow(x4, 2), 3.5));

      vesc = sqrt(2) * pow(1 + pow(r, 2), -0.25);
      vel = vesc / x4;

      theta = (double)rand() / (double)RAND_MAX * M_PI;
      phi = (double)rand() / (double)RAND_MAX * 2 * M_PI;

      V[i] = vel * sin(theta) * cos(phi);
      V[i + 1] = vel * sin(theta) * sin(phi);
      V[i + 2] = vel * cos(theta);
    }

    // starting position
    for (i = 0; i < N * 3; i += 3)
    {
      printf("\nMass %d\tC[x]: %.3f\tC[y]: %.3f\tC[z]: %.3f", i / 3, C[i], C[i + 1], C[i + 2]);
    }

    printf("\n\n");
  }

  // transfering data between processes (tow-side communication)
  MPI_Allgather(&length, 1, MPI_INT, Counts, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&start, 1, MPI_INT, StartPoint, 1, MPI_INT, MPI_COMM_WORLD);

  for (i = 0; i < size; i++)
  {
    StartPoint[i] *= 3;
    EndPoint[i] = StartPoint[i] + Counts[i] - 1;
  }

  // transfering starting data from master to all processes (C,V)
  MPI_Bcast(C, N * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    for (i = 1; i < size; i++)
    {
      for (j = StartPoint[i]; j <= EndPoint[i]; j++)
      {
        MPI_Send(&C[j], 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        MPI_Send(&V[j], 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
      }
    }

    for (i = StartPoint[rank]; i <= EndPoint[rank]; i++)
    {
      Cw[i] = C[i];
      Vw[i] = V[i];
    }
  }

  // processes receiving (C,V)
  if (rank != 0)
  {
    for (j = 0; j < length; j++)
    {
      MPI_Recv(&Cw[j], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&Vw[j], 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  for (i = 0; i < length; i++)
  {
    Fw[i] = 0.0;
  }

  mSquare = pow(M, 2);
  expSquare = pow(Exp, 2);

  // calculations
  for (i = 0; i < loop; i++) // i: stands for the iterations of calculations
  {
    for (j = start; j <= end; j++) // j: body that forces applied to
    {
      for (k = 0; k < N; k++) // k: body that apply force to j-body
      {
        if (j != k)
        {
          dx = C[k * 3 + 0] - C[j * 3 + 0];
          dy = C[k * 3 + 1] - C[j * 3 + 1];
          dz = C[k * 3 + 2] - C[j * 3 + 2];

          d = sqrt(pow(dx, 2) + pow(dy, 2) + pow(dz, 2));
          dSquare = pow(d, 2);

          Fw[(j - start) * 3 + 0] -= G * mSquare * d / (pow(dSquare + expSquare, 1.5) * dx); // total force applies to j-body
          Fw[(j - start) * 3 + 1] -= G * mSquare * d / (pow(dSquare + expSquare, 1.5) * dy); // from all the other k-bodies
          Fw[(j - start) * 3 + 2] -= G * mSquare * d / (pow(dSquare + expSquare, 1.5) * dz); // in the three axis (x,y,z)
        }
      }
    }

    for (j = 0; j < multitude; j++) // j: body that coordinates are calculated for
    {
      for (k = 0; k < 3; k++) // k: axis (x,y,z) that calculated in every loop
      {
        a = Fw[j * 3 + k] / M; // acceleration a=F/M

        Fw[j * 3 + k] = 0.0; // resetting force to zero

        Vw[j * 3 + k] = Vw[j * 3 + k] + a * Dt;                      // velocity V(n+1)=Vn+a*Δt
        Cw[j * 3 + k] = C[(j + start) * 3 + k] + Vw[j * 3 + k] * Dt; // coordinates R(n+1)=Rn+Vn*Δt
      }
    }

    // update C table with new coordinates of the bodies fro the next loop
    MPI_Allgatherv(&Cw[0], length, MPI_DOUBLE, &C[0], Counts, StartPoint, MPI_DOUBLE, MPI_COMM_WORLD);
  }

  if (rank == 0)
  {
    // stop timing
    te = MPI_Wtime();

    // bodies final position
    for (i = 0; i < N * 3; i += 3)
    {
      printf("\nMass %d\tC[x]: %.3f\tC[y]: %.3f\tC[z]: %.3f", i / 3, C[i], C[i + 1], C[i + 2]);
    }
    printf("\n\nTime required is: %.3f\n", te - ts);
  }

  // memory release
  free(Counts);
  free(StartPoint);
  free(EndPoint);

  if (rank == 0)
  {
    free(C);
    free(V);
  }
  free(Cw);
  free(Vw);
  free(Fw);

  MPI_Finalize();

  return 0;
}
