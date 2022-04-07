#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define G 6.673e-11 // gravity constant
#define M 100       // body mass
#define Dt 0.03     // time between iterations
#define Exp 0.01    // softening parameter

int main(int *argc, char **argv)
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

  // C: coordinents table, V: velocity table, F: force table
  double *C, *V, *F;

  // dx/dy/dz: distance in axis (x,y,z), d: distance of two bodies
  double dx, dy, dz, d;

  // dSquare: square of d
  // mSquare: square of body mass
  // expSquare: square of softening parameter
  double dSquare, mSquare, expSquare;

  // a: body acceleration
  double a;

  // ts/te: time mesurement variables
  float ts, te;

  // gather input data from the user
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

  // timing starts
  ts = omp_get_wtime();

  // memory allocation
  C = malloc(N * 3 * sizeof(double));
  F = malloc(N * 3 * sizeof(double));
  V = malloc(N * 3 * sizeof(double));

  if (C == NULL || F == NULL || V == NULL)
  {
    printf("\nAllocation failed\nSimulation aborted");
    return -1;
  }

  // input data initialization
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

  for (i = 0; i < N * 3; i++)
  {
    F[i] = 0.0;
  }

  // starting position
  for (i = 0; i < N * 3; i += 3)
  {
    printf("\nMass %d\tC[x]: %.3f\tC[y]: %.3f\tC[z]: %.3f", i / 3, C[i], C[i + 1], C[i + 2]);
  }

  printf("\n\n");

  mSquare = pow(M, 2);
  expSquare = pow(Exp, 2);

// calculations
#pragma omp parallel private(i, k, dx, dy, dz, d, dSquare, a)
  {
    for (i = 0; i < loop; i++) // i: stands for the iterations of calculations
    {
#pragma omp for schedule(dynamic, 1)
      for (j = 0; j < N; j++) // j: body that forces applied to
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

            F[j * 3 + 0] -= G * mSquare * d / (pow(dSquare + expSquare, 1.5) * dx); // total force applies to j-body
            F[j * 3 + 1] -= G * mSquare * d / (pow(dSquare + expSquare, 1.5) * dy); // from all the other k-bodies
            F[j * 3 + 2] -= G * mSquare * d / (pow(dSquare + expSquare, 1.5) * dz); // in the three axis (x,y,z)
          }
        }
      }
#pragma omp for schedule(dynamic, 1)
      for (j = 0; j < N; j++) // j: body that coordinates are calculated for
      {
        for (k = 0; k < 3; k++) // k: axis (x,y,z) that calculated in every loop
        {
          a = F[j * 3 + k] / M; // acceleration a=F/M

          F[j * 3 + k] = 0.0; // resetting force to zero

          V[j * 3 + k] = V[j * 3 + k] + a * Dt;            // velocity V(n+1)=Vn+a*Δt
          C[j * 3 + k] = C[j * 3 + k] + V[j * 3 + k] * Dt; // coordinates R(n+1)=Rn+Vn*Δt
        }
      }
    }
  }

  // stop timing
  te = omp_get_wtime();

  // bodies final position
  for (i = 0; i < N * 3; i += 3)
  {
    printf("\nMass %d\tC[x]: %.3f\tC[y]: %.3f\tC[z]: %.3f", i / 3, C[i], C[i + 1], C[i + 2]);
  }

  printf("\n\nTime required is: %.3f\n", te - ts);

  // memory release
  free(C);
  free(F);
  free(V);

  return 0;
}
