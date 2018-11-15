#include <string.h>
#include "../mxm.c"
#ifdef _CIVL
#define CBLAS_ORDER int
int CblasRowMajor=101;
int CblasColMajor=102;
#define CBLAS_TRANSPOSE int
int CblasNoTrans=0;
int CblasTrans=1;
int CblasConjTrans=2;
void cblas_dgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double *B, const int ldb,
                 const double beta, double *C, const int ldc) {}
int LAPACK_ROW_MAJOR = 0;
int LAPACKE_dlacpy (int matrix_layout, char uplo, int m, int n, const double* a, int lda, double* b, int ldb) {}
#else
#include <cblas.h>
#include <lapacke.h>
#endif

/*
  'Matrix-tensor-matrix-vector' product 
    vec(V) = alpha * (A \otimes B) vec(U) + beta * vec(V)  <=>  V = alpha * B * U * A^T + beta * V
*/
void mtmv(int m,double alpha,double A[m][m],double B[m][m],double U[m][m],double beta,double V[m][m])
{
  double tmp[m][m],one = 1.,zero = 0.;

  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,one,&B[0][0],m,&U[0][0],m,zero,&tmp[0][0],m);
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&tmp[0][0],m,&A[0][0],m,beta,&V[0][0],m);
}

/*-------------------------------
       FORWARD DERIVATIVES
  ------------------------------- */

/*
  Re-interpretation of Tapenade forward source transformation in terms of zeroout, transpose and mxm
  functions alone. Differentiating w.r.t. both matrix arguments, the product rule gives
    C = A * B  ==>  Cd = Ad * B + A * Bd
*/
void mxm_dot(int m,int p,int n,double A[m][p],double Ad[m][p],double B[p][n],double Bd[p][n],double C[m][n],double Cd[m][n])
{
  /* Undifferentiated function call */
  naive_mxm(m,p,n,A,B,C);

  /* Differentiated function call */
  zeroout(m,n,Cd);
  naive_mxm(m,p,n,Ad,B,Cd);
  naive_mxm(m,p,n,A,Bd,Cd);
}

/*
  Forward mode Hadamard product w.r.t. both matrix arguments
*/
void mpm_dot(int m,int n,double A[m][n],double Ad[m][n],double B[m][n],double Bd[m][n],double C[m][n],double Cd[m][n])
{
  /* Undifferentiated function call */
  naive_mpm(m,n,A,B,C);

  /* Differentiated function call */
  zeroout(m,n,Cd);
  naive_mpm(m,n,Ad,B,Cd);
  naive_mpm(m,n,A,Bd,Cd);
}

/*
  Forward mode square dgemm w.r.t. both matrix arguments
*/
void dgemm_dot(bool transa,bool transb,int m,double alpha,double A[m][m],double Ad[m][m],double B[m][m],double Bd[m][m],double beta,double C[m][m],double Cd[m][m])
{
  double one = 1.,zero = 0.;

  if (!transa) {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    }
  }
}

/*
  Forward mode square dgemm w.r.t. both matrix arguments using naive dgemm
*/
void naive_dgemm_dot(bool transa,bool transb,int m,double alpha,double A[m][m],double Ad[m][m],double B[m][m],double Bd[m][m],double beta,double C[m][m],double Cd[m][m])
{
  double one = 1.,zero = 0.;

  if (!transa) {
    if (!transb) {
      naive_dgemm(CblasNoTrans,CblasNoTrans,m,alpha,A,B,beta,C);
      naive_dgemm(CblasNoTrans,CblasNoTrans,m,alpha,Ad,B,zero,Cd);
      naive_dgemm(CblasNoTrans,CblasNoTrans,m,alpha,A,Bd,one,Cd);
    } else {
      naive_dgemm(CblasNoTrans,CblasTrans,m,alpha,A,B,beta,C);
      naive_dgemm(CblasNoTrans,CblasTrans,m,alpha,Ad,B,zero,Cd);
      naive_dgemm(CblasNoTrans,CblasTrans,m,alpha,A,Bd,one,Cd);
    }
  } else {
    if (!transb) {
      naive_dgemm(CblasTrans,CblasNoTrans,m,alpha,A,B,beta,C);
      naive_dgemm(CblasTrans,CblasNoTrans,m,alpha,Ad,B,zero,Cd);
      naive_dgemm(CblasTrans,CblasNoTrans,m,alpha,A,Bd,one,Cd);
    } else {
      naive_dgemm(CblasTrans,CblasTrans,m,alpha,A,B,beta,C);
      naive_dgemm(CblasTrans,CblasTrans,m,alpha,Ad,B,zero,Cd);
      naive_dgemm(CblasTrans,CblasTrans,m,alpha,A,Bd,one,Cd);
    }
  }
}

/*
  Forward mode square dgemm w.r.t first matrix argument
*/
void dgemm_A_dot(bool transa,bool transb,int m,double alpha,double A[m][m],double Ad[m][m],double B[m][m],double beta,double C[m][m],double Cd[m][m])
{
  double zero = 0.;

  if (!transa) {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    }
  }
}

/*
  Forward mode square dgemm w.r.t second matrix argument
*/
void dgemm_B_dot(bool transa,bool transb,int m,double alpha,double A[m][m],double B[m][m],double Bd[m][m],double beta,double C[m][m],double Cd[m][m])
{
  double zero = 0.;

  if (!transa) {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    }
  }
}

/*
  Forward mode square dgemm w.r.t. both scalar arguments using naive dgemm
*/
void naive_dgemm_scalar_dot(bool transa,bool transb,int m,double alpha,double alphad,double A[m][m],double B[m][m],double beta,double betad,double C[m][m],double Cd[m][m])
{
  double one = 1.,zero = 0.;
  char   all[1] = "A";

  if (!transa) {
    if (!transb) {
      memcpy(&Cd[0][0], &C[0][0], m*m*sizeof(double));
      naive_dgemm(CblasNoTrans,CblasNoTrans,m,alpha,A,B,beta,C);
      naive_dgemm(CblasNoTrans,CblasNoTrans,m,alphad,A,B,betad,Cd);
    } else {
      memcpy(&Cd[0][0], &C[0][0], m*m*sizeof(double));
      naive_dgemm(CblasNoTrans,CblasTrans,m,alpha,A,B,beta,C);
      naive_dgemm(CblasNoTrans,CblasTrans,m,alphad,A,B,betad,Cd);
    }
  } else {
    if (!transb) {
      memcpy(&Cd[0][0], &C[0][0], m*m*sizeof(double));
      naive_dgemm(CblasTrans,CblasNoTrans,m,alpha,A,B,beta,C);
      naive_dgemm(CblasTrans,CblasNoTrans,m,alphad,A,B,betad,Cd);
    } else {
      memcpy(&Cd[0][0], &C[0][0], m*m*sizeof(double));
      naive_dgemm(CblasTrans,CblasTrans,m,alpha,A,B,beta,C);
      naive_dgemm(CblasTrans,CblasTrans,m,alphad,A,B,betad,Cd);
    }
  }
}

/*
  Forward mode square dgemm w.r.t. both scalar arguments
*/
void dgemm_scalar_dot(bool transa,bool transb,int m,double alpha,double alphad,double A[m][m],double B[m][m],double beta,double betad,double C[m][m],double Cd[m][m])
{
  double one = 1.,zero = 0.;
  char   all[1] = "A";

  if (!transa) {
    if (!transb) {
      LAPACKE_dlacpy(LAPACK_ROW_MAJOR,all[0],m,m,&C[0][0],m,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alphad,&A[0][0],m,&B[0][0],m,betad,&Cd[0][0],m);
    } else {
      LAPACKE_dlacpy(LAPACK_ROW_MAJOR,all[0],m,m,&C[0][0],m,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alphad,&A[0][0],m,&B[0][0],m,betad,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      LAPACKE_dlacpy(LAPACK_ROW_MAJOR,all[0],m,m,&C[0][0],m,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alphad,&A[0][0],m,&B[0][0],m,betad,&Cd[0][0],m);
    } else {
      LAPACKE_dlacpy(LAPACK_ROW_MAJOR,all[0],m,m,&C[0][0],m,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alphad,&A[0][0],m,&B[0][0],m,betad,&Cd[0][0],m);
    }
  }
}

/*
  Forward mode matrix-tensor-matrix-vector product w.r.t. both matrix arguments
*/
void mtmv_dot(int m,double alpha,double A[m][m],double B[m][m],double U[m][m],double Ud[m][m],double beta,double V[m][m],double Vd[m][m])
{
  double tmp[m][m],tmpd[m][m],one = 1.,zero = 0.;

  zeroout(m,m,Vd);
  dgemm_B_dot(0,0,m,one,B,U,Ud,zero,tmp,tmpd);
  dgemm_A_dot(0,1,m,alpha,tmp,tmpd,A,beta,V,Vd);
}

/*-------------------------------
       REVERSE DERIVATIVES
  ------------------------------- */

/*
  Re-interpretation of Tapenade reverse source transformation in terms of zeroout, transpose and mxm
  functions alone. Differentiating w.r.t. both matrix arguments,
    C = A * B  ==>  Ab = Cb * B^T, Bb = A^T * Cb
*/
void mxm_bar(int m,int p,int n,double A[m][p],double Ab[m][p],double B[p][n],double Bb[p][n],double C[m][n],double Cb[m][n])
{
  double At[p][m],Bt[n][p];

  zeroout(m,p,Ab);
  zeroout(p,n,Bb);
  transpose(m,p,A,At);
  transpose(p,n,B,Bt);
  naive_mxm(m,n,p,Cb,Bt,Ab);
  naive_mxm(p,m,n,At,Cb,Bb);
}

/*
  Reverse mode Hadamard product w.r.t. both matrix arguments
*/
void mpm_bar(int m,int n,double A[m][n],double Ab[m][n],double B[m][n],double Bb[m][n],double C[m][n],double Cb[m][n])
{
  zeroout(m,n,Ab);
  zeroout(m,n,Bb);
  naive_mpm(m,n,Cb,B,Ab);
  naive_mpm(m,n,A,Cb,Bb);
}

/*
  Reverse mode square dgemm w.r.t. both matrix arguments
*/
void dgemm_bar(bool transa,bool transb,int m,double alpha,double A[m][m],double Ab[m][m],double B[m][m],double Bb[m][m],double beta,double C[m][m],double Cb[m][m])
{
  double zero = 0.;

  if (!transa) {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&Cb[0][0],m,&B[0][0],m,zero,&Ab[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Cb[0][0],m,zero,&Bb[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&Cb[0][0],m,&B[0][0],m,zero,&Ab[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&Cb[0][0],m,&A[0][0],m,zero,&Bb[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&B[0][0],m,&Cb[0][0],m,zero,&Ab[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Cb[0][0],m,zero,&Bb[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&B[0][0],m,&Cb[0][0],m,zero,&Ab[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&Cb[0][0],m,&A[0][0],m,zero,&Bb[0][0],m);
    }
  }
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,zero,&A[0][0],m,&B[0][0],m,beta,&Cb[0][0],m);
}

/*
  Reverse mode square dgemm w.r.t. both matrix arguments using naive dgemm
*/
void naive_dgemm_bar(bool transa,bool transb,int m,double alpha,double A[m][m],double Ab[m][m],double B[m][m],double Bb[m][m],double beta,double C[m][m],double Cb[m][m])
{
  double zero = 0.;

  if (!transa) {
    if (!transb) {
      naive_dgemm(CblasNoTrans,CblasTrans,m,alpha,Cb,B,zero,Ab);
      naive_dgemm(CblasTrans,CblasNoTrans,m,alpha,A,Cb,zero,Bb);
    } else {
      naive_dgemm(CblasNoTrans,CblasNoTrans,m,alpha,Cb,B,zero,Ab);
      naive_dgemm(CblasTrans,CblasNoTrans,m,alpha,Cb,A,zero,Bb);
    }
  } else {
    if (!transb) {
      naive_dgemm(CblasNoTrans,CblasTrans,m,alpha,B,Cb,zero,Ab);
      naive_dgemm(CblasNoTrans,CblasNoTrans,m,alpha,A,Cb,zero,Bb);
    } else {
      naive_dgemm(CblasTrans,CblasTrans,m,alpha,B,Cb,zero,Ab);
      naive_dgemm(CblasTrans,CblasTrans,m,alpha,Cb,A,zero,Bb);
    }
  }
  naive_dgemm(CblasNoTrans,CblasNoTrans,m,zero,A,B,beta,Cb);
}

/*
  Reverse mode square dgemm w.r.t. first matrix argument

  NOTE: A can be NULL
*/
void dgemm_A_bar(bool transa,bool transb,int m,double alpha,double A[m][m],double Ab[m][m],double B[m][m],double beta,double C[m][m],double Cb[m][m])
{
  double zero = 0.;

  if (!transa) {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&Cb[0][0],m,&B[0][0],m,zero,&Ab[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&Cb[0][0],m,&B[0][0],m,zero,&Ab[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&B[0][0],m,&Cb[0][0],m,zero,&Ab[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&B[0][0],m,&Cb[0][0],m,zero,&Ab[0][0],m);
    }
  }
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,zero,&A[0][0],m,&B[0][0],m,beta,&Cb[0][0],m);
}

/*
  Reverse mode square dgemm w.r.t. second matrix argument

  NOTE: B can be NULL
*/
void dgemm_B_bar(bool transa,bool transb,int m,double alpha,double A[m][m],double B[m][m],double Bb[m][m],double beta,double C[m][m],double Cb[m][m])
{
  double zero = 0.;

  if (!transa) {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Cb[0][0],m,zero,&Bb[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&Cb[0][0],m,&A[0][0],m,zero,&Bb[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Cb[0][0],m,zero,&Bb[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&Cb[0][0],m,&A[0][0],m,zero,&Bb[0][0],m);
    }
  }
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,zero,&A[0][0],m,&B[0][0],m,beta,&Cb[0][0],m);
}

/*
  Reverse mode square dgemm w.r.t. both scalar arguments

  NOTE: C can be NULL
*/
void dgemm_s_bar(bool transa,bool transb,int m,double alpha,double alphab,double A[m][m],double B[m][m],double beta,double betab,double C[m][m],double Cb[m][m])
{
  double zero = 0.,tmp[m][m];

  if (!transa) {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&tmp[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&tmp[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&tmp[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&tmp[0][0],m);
    }
  }
  naive_fmpm(m,m,Cb,tmp,alphab);
  naive_fmpm(m,m,Cb,C,betab);
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,zero,&A[0][0],m,&B[0][0],m,beta,&Cb[0][0],m);
}


/*
  Reverse mode matrix-tensor-matrix-vector product
*/
void mtmv_bar(int m,double alpha,double A[m][m],double B[m][m],double U[m][m],double Ub[m][m],double beta,double V[m][m],double Vb[m][m])
{
  double tmpb[m][m],one = 1;

  dgemm_A_bar(0,1,m,alpha,NULL,tmpb,A,beta,V,Vb);
  dgemm_B_bar(0,0,m,1,B,U,Ub,1,NULL,tmpb);
}
