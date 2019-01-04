#include <stdio.h>
#include <stdbool.h>

typedef struct {
  double u,v;
} Field;

/*
  Incremental matrix-matrix multiply

  C = C + A*B
*/
void naive_mxm(int m,int p,int n,double **A,double **B,double **C)
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
  Incremental vector addition with scaling

  y = y + a*x
*/
void naive_axpy(int n,double a,double *x,double *y)
{
  int i;

  for (i=0; i<n; i++) {
    if (a == 1.)
      y[i] += x[i];
    else if (a == 0.)
      y[i] = 0.;
    else
      y[i] += a*x[i];
  }
}

/*
  Incremental matrix addition on square matrices with scaling

  y = y + a*x
*/
void naive_matrix_axpy(int n,double a,double **X,double **Y)
{
  int i,j;

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      if (a == 1.)
        Y[i][j] += X[i][j];
      else if (a == 0.)
        Y[i][j] = 0.;
      else
        Y[i][j] += a*X[i][j];
    }
  }
}

/*
  Incremental pointwise matrix multiply (Hadamard product)

  C = C + A.*B
*/
void naive_mpm(int m,int n,double **A,double **B,double **C)
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      C[i][j] += A[i][j] * B[i][j];
    }
  }
}

/*
  Incremental Frobenius norm of Hadamard product

  c = c + ||A.*B||_F
*/
void naive_fmpm(int m,int n,double **A,double **B,double c)
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      c += A[i][j] * B[i][j];
    }
  }
}

/*
  Simplest implementation of dgemm for square matrices
*/
void extra_naive_dgemm(bool transa,bool transb,int m,double alpha,double **A,double **B,double beta,double **C)
{
  int i,j,k;

  for (i=0; (i<m); i++) {
    for (j=0; (j<m); j++) {
      C[i][j] = beta * C[i][j];
    }
  }
  if (!transa) {
    if (!transb) {
      for (i=0; i<m; i++) {
        for (j=0; j<m; j++) {
          for (k=0; k<m; k++) {
            C[i][j] += alpha * A[i][k] * B[k][j];
          }
        }
      }
    } else {
      for (i=0; i<m; i++) {
        for (j=0; j<m; j++) {
          for (k=0; k<m; k++) {
            C[i][j] += alpha * A[i][k] * B[j][k];
          }
        }
      }
    }
  } else {
    if (!transb) {
      for (i=0; i<m; i++) {
        for (j=0; j<m; j++) {
          for (k=0; k<m; k++) {
            C[i][j] += alpha * A[k][i] * B[k][j];
          }
        }
      }
    } else {
      for (i=0; i<m; i++) {
        for (j=0; j<m; j++) {
          for (k=0; k<m; k++) {
            C[i][j] += alpha * A[k][i] * B[j][k];
          }
        }
      }
    }
  }
}

/*
  Basic implementation of dgemm for square matrices
*/
void naive_dgemm(bool transa,bool transb,int m,double alpha,double **A,double **B,double beta,double **C)
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
  Double Kronecker product (on square matrices)

  vec(V) = (A \otimes B) vec(U)  <=>  V = B * U * A^T
*/
void naive_mtmv(int m,double alpha,double **A,double **B,double **U,double beta,double **V)
{
  int    i,j,k;
  double **tmp = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) tmp[i] = malloc(m*sizeof(double));

  for (i=0; (i<m); i++) {
    for (j=0; (j<m); j++) {
      tmp[i][j] = 0.;
      if (beta == 0.) {
        V[i][j] = 0.;
      } else if (beta != 1.) {
        V[i][j] = beta * V[i][j];
      }
    }
  }
  for (i=0; i<m; i++) {
    for (j=0; j<m; j++) {
      for (k=0; k<m; k++) {
        tmp[i][j] += B[i][k] * U[k][j];
      }
    }
  }
  for (i=0; i<m; i++) {
    for (j=0; j<m; j++) {
      for (k=0; k<m; k++) {
        if (alpha == 1.) {
          V[i][j] += tmp[i][k] * A[j][k];
        } else {
          V[i][j] += alpha * tmp[i][k] * A[j][k];
        }
      }
    }
  }

  for (i=0; i<m; i++) free(tmp[i]);
  free(tmp);
}

/*
  Double Kronecker product (as above) using double applications of naive_dgemm
*/
void extra_naive_mtmv(int m,double alpha,double **A,double **B,double **U,double beta,double **V)
{
  double one = 1.,zero = 0.;
  int i;
  double **tmp = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) tmp[i] = malloc(m*sizeof(double));

  naive_dgemm(0,0,m,one,B,U,zero,tmp);
  naive_dgemm(0,1,m,alpha,tmp,A,beta,V);

  for (i=0; i<m; i++) free(tmp[i]);
  free(tmp);
}

/*
  Zero array entries
*/
void zeroout(int m,int n,double **A)
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
void transpose(int m,int n,double **A,double **At)
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
void printmat(const char *name,int m,int n,double **A)
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
