#include "mxm.c"

/*-------------------------------
       FORWARD DERIVATIVES
  ------------------------------- */

/*
  Re-interpretation of Tapenade forward source transformation in terms of zeroout, transpose and mxm
  functions alone. By the product rule
    C = A * B  ==>  Cd = Ad * B + A * Bd 
*/
void mxm_forward(int m,int p,int n,double A[m][p],double Ad[m][p],double B[p][n],double Bd[p][n],double C[m][n],double Cd[m][n])
{
  /* Undifferentiated function call */
  mxm(m,p,n,A,B,C);

  /* Differentiated function call */
  zeroout(m,n,Cd);
  mxm(m,p,n,Ad,B,Cd);
  mxm(m,p,n,A,Bd,Cd);
}

/*
  Pointwise matrix multiplication
*/
void mpm_forward(int m,int n,double A[m][n],double Ad[m][n],double B[m][n],double Bd[m][n],double C[m][n],double Cd[m][n])
{
  /* Undifferentiated function call */
  mpm(m,n,A,B,C);

  /* Differentiated function call */
  zeroout(m,n,Cd);
  mpm(m,n,Ad,B,Cd);
  mpm(m,n,A,Bd,Cd);
}

/*-------------------------------
       REVERSE DERIVATIVES
  ------------------------------- */

/*
  Re-interpretation of Tapenade reverse source transformation in terms of zeroout, transpose and mxm
  functions alone. By the product rule
    C = A * B  ==>  Ab = Cb * B^T, Bb = A^T * Cb
*/
void mxm_reverse(int m,int p,int n,double A[m][p],double Ab[m][p],double B[p][n],double Bb[p][n],double C[m][n],double Cb[m][n])
{
  double At[p][m],Bt[n][p];

  zeroout(m,p,Ab);
  zeroout(p,n,Bb);
  transpose(m,p,A,At);
  transpose(p,n,B,Bt);
  mxm(m,n,p,Cb,Bt,Ab);
  mxm(p,m,n,At,Cb,Bb);
}

void mpm_reverse(int m,int n,double A[m][n],double Ab[m][n],double B[m][n],double Bb[m][n],double C[m][n],double Cb[m][n])
{
  zeroout(m,n,Ab);
  zeroout(m,n,Bb);
  mpm(m,n,Cb,B,Ab);
  mpm(m,n,A,Cb,Bb);
}
