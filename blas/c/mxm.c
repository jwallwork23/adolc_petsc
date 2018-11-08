#include <stdio.h>
#include <stdbool.h>

/*
  Incremental matrix-matrix multiply

  C = C + A*B
*/
void naive_mxm(int m,int p,int n,double A[m][p],double B[p][n],double C[m][n])
{
  int i,j,k;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      for (k=0; k<p; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}

/*
  Basic implementation of dgemm for square matrices
*/
void naive_dgemm(bool transa,bool transb,int m,double alpha,double A[m][m],double B[m][m],double beta,double C[m][m])
{
  int i,j,k;

  for (i=0; (i<m); i++) {
    for (j=0; (j<m); j++) {
      if (beta == 0.) {
        C[i][j] = 0.;
      } else if (beta != 1.) {
        C[i][j] = beta * C[i][j];
      }
    }
  }
  if (alpha != 0.) {
    if (!transa) {
      if (!transb) {
        for (i=0; i<m; i++) {
          for (j=0; j<m; j++) {
            for (k=0; k<m; k++) {
              if (alpha == 1.) {
                C[i][j] += A[i][k] * B[k][j];
              } else {
                C[i][j] += alpha * A[i][k] * B[k][j];
              }
            }
          }
        }
      } else {
        for (i=0; i<m; i++) {
          for (j=0; j<m; j++) {
            for (k=0; k<m; k++) {
              if (alpha == 1.) {
                C[i][j] += A[i][k] * B[j][k];
              } else {
                C[i][j] += alpha * A[i][k] * B[j][k];
              }
            }
          }
        }
      }
    } else {
      if (!transb) {
        for (i=0; i<m; i++) {
          for (j=0; j<m; j++) {
            for (k=0; k<m; k++) {
              if (alpha == 1.) {
                C[i][j] += A[k][i] * B[k][j];
              } else {
                C[i][j] += alpha * A[k][i] * B[k][j];
              }
            }
          }
        }
      } else {
        for (i=0; i<m; i++) {
          for (j=0; j<m; j++) {
            for (k=0; k<m; k++) {
              if (alpha == 1.) {
                C[i][j] += A[k][i] * B[j][k];
              } else {
                C[i][j] += alpha * A[k][i] * B[j][k];
              }
            }
          }
        }
      }
    }
  }
}

/*
  Incremental pointwise matrix multiply (Hadamard product)

  C = C + A.*B
*/
void naive_mpm(int m,int n,double A[m][n],double B[m][n],double C[m][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      C[i][j] += A[i][j] * B[i][j];
    }
  }
}

/*
  Zero array entries
*/
void zeroout(int m,int n,double A[m][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      A[i][j] = 0;
    }
  }
}

/*
  Matrix transpose
*/
void transpose(int m,int n,double A[m][n],double At[n][m])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      At[j][i] = A[i][j];
    }
  }
}

/*
  Simple matrix printing
*/
void printmat(const char *name,int m,int n,double A[m][n])
{
  int i,j;

  printf("%s\n",name);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      printf("%.1f,",A[i][j]);
    }
    printf("\n");
  }
}
