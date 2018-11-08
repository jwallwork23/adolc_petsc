#include <stdbool.h>

/*
  Basic implementation of dgemm for square matrices  TODO: hard-code this into the below
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
  Double Kronecker product (on square matrices)

  vec(V) = (A \otimes B) vec(U)  <=>  V = B * U * A^T
*/
void naive_mtmv(int m,double alpha,double A[m][m],double B[m][m],double U[m][m],double beta,double V[m][m])
{
  double tmp[m][m],one = 1.;

  naive_dgemm(0,0,m,one,B,U,one,tmp);
  naive_dgemm(0,1,m,alpha,tmp,A,beta,V);
}
