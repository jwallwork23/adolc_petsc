#include "../mxm.c"
#include <cblas.h>


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
  functions alone. By the product rule
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
  Forward mode Hadamard product
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
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
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
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
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
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    }
  }
}

/*
  Forward mode double Kronecker product TODO: test
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
  functions alone. By the product rule
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
  Reverse mode Hadamard product
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
  Reverse mode square dgemm w.r.t. first matrix argument
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
  Reverse mode double Kronecker product TODO: test
*/
void mtmv_bar(int m,double alpha,double A[m][m],double B[m][m],double U[m][m],double Ub[m][m],double beta,double V[m][m],double Vb[m][m])
{
  double tmp[m][m],tmpb[m][m],one = 1;

  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,one,&B[0][0],m,&U[0][0],m,one,&tmp[0][0],m);
  dgemm_A_bar(0,1,m,alpha,NULL,tmpb,A,beta,V,Vb);
  dgemm_B_bar(0,0,m,1,B,U,Ub,1,NULL,tmpb);
}

// TODO: Tapenade efficiency is dependent on ordering in mtmv  - interesting note for paper.
