#include "../mxm.c"

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
  Forward mode simple dgemm
*/
void dgemm_dot(bool transa,bool transb,int m,double A[m][m],double Ad[m][m],double B[m][m],double Bd[m][m],double C[m][m],double Cd[m][m])
{
  /* Undifferentiated function call */
  dgemm(transa,transa,m,A,B,C);

  /* Differentiated function call */
  zeroout(m,m,Cd);
  dgemm(transa,transb,m,Ad,B,Cd);
  dgemm(transa,transb,m,A,Bd,Cd);
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
  Reverse mode simple dgemm
*/
void dgemm_bar(bool transa,bool transb,int m,double A[m][m],double Ab[m][m],double B[m][m],double Bb[m][m],double C[m][m],double Cb[m][m])
{
  zeroout(m,m,Ab);
  zeroout(m,m,Bb);
  if (!transa)
    dgemm(0,!transb,m,Cb,B,Ab);
  else
    dgemm(transb,1,m,B,Cb,Ab);
  if (!transb)
    dgemm(!transa,0,m,A,Cb,Bb);
  else
    dgemm(1,transa,m,Cb,A,Bb);
}

