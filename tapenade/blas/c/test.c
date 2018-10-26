#include <assert.h>
#include <math.h>
#include "derivatives.c"


void inittest(int m,int p,int n,double A[m][p],double B[p][n]);
void initforward(int m,int p,int n,double Ad[m][p],double Bd[p][n]);
void initreverse(int m,int n,double Cb[m][n]);

void checkval(double approx,double exact);
void checkmult(double C[2][2]);
void checkforward(double Cd[2][2]);
void checkreverse(double Ab[2][3],double Bb[3][2]);

int main()
{
  int    m = 2,p=3,n=2;
  double A[m][p],B[p][n],C[m][n];
  double Ad[m][p],Bd[p][n],Cd[m][n];
  double Ab[m][p],Bb[p][n],Cb[m][n];

  /* TEST 1: matrix-matrix multiply */

  inittest(m,p,n,A,B);
  zeroout(m,n,C);
  mxm(m,p,n,A,B,C);
  checkmult(C);

  /* TEST 2: forward mode */

  inittest(m,p,n,A,B);
  initforward(m,p,n,Ad,Bd);
  zeroout(m,n,C);
  zeroout(m,n,Cd);
  mxm_forward(m,p,n,A,Ad,B,Bd,C,Cd);
  checkforward(Cd);

  /* TEST 3: reverse mode */

  inittest(m,p,n,A,B);
  initreverse(m,n,Cb);
  zeroout(m,p,Ab);
  zeroout(p,n,Bb);
  mxm_reverse(m,p,n,A,Ab,B,Bb,C,Cb);
  checkreverse(Ab,Bb);

  printf("All tests passed.\n");

  return 0;
}

void checkval(double approx,double exact)
{
  assert(fabs(approx-exact)<1e-8);
}

void inittest(int m,int p,int n,double A[m][p],double B[p][n])
{
  A[0][0] = 1.;A[0][1] = 2.;A[0][2] = 3.;
  A[1][0] = 0.;A[1][1] = 1.;A[1][2] = 2.;

  B[0][0] = 0.;B[0][1] = -1.;
  B[1][0] = 1.;B[1][1] = 0.;
  B[2][0] = 2.;B[2][1] = 1.;
}

void checkmult(double C[2][2])
{
  checkval(C[0][0],8.);checkval(C[0][1],2.);
  checkval(C[1][0],5.);checkval(C[1][1],2.);
}

void initforward(int m,int p,int n,double Ad[m][p],double Bd[p][n])
{
  Ad[0][0] = 1.;Ad[0][1] = 0.;Ad[0][2] = 0.;
  Ad[1][0] = 0.;Ad[1][1] = 1.;Ad[1][2] = 0.;

  Bd[0][0] = 1.;Bd[0][1] = 0.;
  Bd[1][0] = 0.;Bd[1][1] = 1.;
  Bd[2][0] = 0.;Bd[2][1] = 0.;
}

void checkforward(double Cd[2][2])
{
  checkval(Cd[0][0],1.);checkval(Cd[0][1],1.);
  checkval(Cd[1][0],1.);checkval(Cd[1][1],1.);
}

void initreverse(int m,int n,double Cb[m][n])
{
  Cb[0][0] = 1.;Cb[0][1] = -1.;
  Cb[1][0] = -1.;Cb[1][1] = 1.;
}

void checkreverse(double Ab[2][3],double Bb[3][2])
{
  checkval(Ab[0][0],1.);checkval(Ab[0][1],1.);checkval(Ab[0][2],1.);
  checkval(Ab[1][0],-1.);checkval(Ab[1][1],-1.);checkval(Ab[1][2],-1.);

  checkval(Bb[0][0],1.);checkval(Bb[0][1],-1.);
  checkval(Bb[1][0],1.);checkval(Bb[1][1],-1.);
  checkval(Bb[2][0],1.);checkval(Bb[2][1],-1.);
}
