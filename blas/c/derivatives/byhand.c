#include "../mxm.c"
#include <cblas.h>

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
  mxm(m,p,n,A,B,C);

  /* Differentiated function call */
  zeroout(m,n,Cd);
  mxm(m,p,n,Ad,B,Cd);
  mxm(m,p,n,A,Bd,Cd);
}

/*
  Forward mode Hadamard product
*/
void mpm_dot(int m,int n,double A[m][n],double Ad[m][n],double B[m][n],double Bd[m][n],double C[m][n],double Cd[m][n])
{
  /* Undifferentiated function call */
  mpm(m,n,A,B,C);

  /* Differentiated function call */
  zeroout(m,n,Cd);
  mpm(m,n,Ad,B,Cd);
  mpm(m,n,A,Bd,Cd);
}

/*
  Forward mode square dgemm w.r.t. both matrix arguments
*/
void dgemm_dot(bool transa,bool transb,int m,double alpha,double A[m][m],double Ad[m][m],double B[m][m],double Bd[m][m],double beta,double C[m][m],double Cd[m][m])
{
  double one = 1.,zero = 0.;

  if (!transa) {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,one,&Cd[0][0],m);
    } else {
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
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&Ad[0][0],m,&B[0][0],m,zero,&Cd[0][0],m);
    } else {
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
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    }
  } else {
    if (!transb) {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    } else {
      cblas_dgemm(CblasRowMajor,CblasTrans,CblasTrans,m,m,m,alpha,&A[0][0],m,&Bd[0][0],m,zero,&Cd[0][0],m);
    }
  }
}

/*
  Forward mode double Kronecker product w.r.t. both matrix arguments
*/
void mtmv_dot(int m,double alpha,double A[m][m],double Ad[m][m],double B[m][m],double Bd[m][m],double U[m][m],double beta,double V[m][m],double Vd[m][m])
{
  double tmp[m][m],tmpd[m][m],one = 1;

  dgemm_A_dot(0,0,m,one,B,Bd,U,one,tmp,tmpd);
  dgemm_dot(0,1,m,alpha,tmp,tmpd,A,Ad,beta,V,Vd);
}

/*
  Forward mode double Kronecker product w.r.t. first matrix argument
*/
void mtmv_A_dot(int m,double alpha,double A[m][m],double Ad[m][m],double B[m][m],double U[m][m],double beta,double V[m][m],double Vd[m][m])
{
  double tmp[m][m],one = 1,zero = 0.;

  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,one,&B[0][0],m,&U[0][0],m,zero,&tmp[0][0],m);
  dgemm_B_dot(0,1,m,alpha,tmp,A,Ad,one,V,Vd);
}

/*
  Forward mode double Kronecker product w.r.t. second matrix argument
*/
void mtmv_B_dot(int m,double alpha,double A[m][m],double B[m][m],double Bd[m][m],double U[m][m],double beta,double V[m][m],double Vd[m][m])
{
  double tmp[m][m],one = 1,zero = 0.;

  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,m,one,&U[0][0],m,&A[0][0],m,zero,&tmp[0][0],m);
  dgemm_A_dot(0,0,m,alpha,B,Bd,tmp,beta,V,Vd);
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
  mxm(m,n,p,Cb,Bt,Ab);
  mxm(p,m,n,At,Cb,Bb);
}

/*
  Reverse mode Hadamard product
*/
void mpm_bar(int m,int n,double A[m][n],double Ab[m][n],double B[m][n],double Bb[m][n],double C[m][n],double Cb[m][n])
{
  zeroout(m,n,Ab);
  zeroout(m,n,Bb);
  mpm(m,n,Cb,B,Ab);
  mpm(m,n,A,Cb,Bb);
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
  Reverse mode double Kronecker product w.r.t. first two matrix arguments
*/
void mtmv_bar(int m,double alpha,double A[m][m],double Ab[m][m],double B[m][m],double Bb[m][m],double U[m][m],double beta,double V[m][m],double Vb[m][m])
{
  double tmp[m][m],tmpb[m][m],one = 1;
  //double Ub[m][m];

  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,one,&B[0][0],m,&U[0][0],m,one,&tmp[0][0],m);
  zeroout(m,m,tmpb);
  dgemm_bar(0,1,m,alpha,tmp,tmpb,A,Ab,beta,V,Vb);
  //zeroout(m,m,Ub);
  //dgemm_bar(0,0,m,1,B,Bb,U,Ub,1,tmp,tmpb);
  dgemm_A_bar(0,0,m,1,B,Bb,U,1,tmp,tmpb);
}

/*
  Reverse mode double Kronecker product w.r.t. first matrix argument
*/
void mtmv_A_bar(int m,double alpha,double A[m][m],double Ab[m][m],double B[m][m],double U[m][m],double beta,double V[m][m],double Vb[m][m])
{
  double tmp[m][m],one = 1;
  //double tmpb[m][m];

  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,one,&B[0][0],m,&U[0][0],m,one,&tmp[0][0],m);
  //dgemm_bar(0,1,m,alpha,tmp,tmpb,A,Ab,beta,V,Vb);
  dgemm_A_bar(0,1,m,alpha,tmp,A,Ab,beta,V,Vb);
}

/*
  Reverse mode double Kronecker product w.r.t. second matrix argument
*/
void mtmv_B_bar(int m,double alpha,double A[m][m],double B[m][m],double Bb[m][m],double U[m][m],double beta,double V[m][m],double Vb[m][m])
{
  double tmp[m][m],one = 1,zero = 0.;

  dgemm_A_bar(0,1,m,alpha,tmp,tmpb,A,beta,V,Vb);
  dgemm_A_bar(0,0,m,1,B,Bb,U,1,tmp,tmpb);

  // TODO: If you reorder these then you get more efficiency - interesting note for paper.
}
