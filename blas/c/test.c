#include <assert.h>
#include <math.h>
#include "derivatives.c"
#include <time.h>
#include "mxm_d.c"
#include "mxm_b.c"


void inittest(int m,int p,int n,double A[m][p],double B[p][n]);
void initforward(int m,int p,int n,double Ad[m][p],double Bd[p][n]);
void initreverse(int m,int n,double Cb[m][n]);

void checkval(double approx,double exact);
void checkmult(int m,int n,double C[m][n]);
void checkforward(int m,int n,double Cd[m][n]);
void checkreverse(int m,int p,int n,double Ab[m][p],double Bb[p][n]);

int main()
{
  clock_t t;
  int     m = 2,p=3,n=2;
  double  A[m][p],B[p][n],C[m][n];
  double  Ad[m][p],Bd[p][n],Cd[m][n];
  double  Ab[m][p],Bb[p][n],Cb[m][n];

  /* TEST 1: matrix-matrix multiply */

  inittest(m,p,n,A,B);
  zeroout(m,n,C);
  t = clock();
  mxm(m,p,n,A,B,C);
  t = clock() - t;
  checkmult(m,n,C);
  printf("Original mxm call: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);

  /* TEST 2: forward mode with Tapenade */

  inittest(m,p,n,A,B);
  initforward(m,p,n,Ad,Bd);
  zeroout(m,n,C);
  t = clock();
  mxm_d(m,p,n,A,Ad,B,Bd,C,Cd);
  t = clock() - t;
  checkforward(m,n,Cd);
  printf("Forward mode with Tapenade: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);

  /* TEST 3: forward mode with mxms */

  inittest(m,p,n,A,B);
  initforward(m,p,n,Ad,Bd);
  zeroout(m,n,C);
  t = clock();
  zeroout(m,n,Cd);
  mxm_forward(m,p,n,A,Ad,B,Bd,C,Cd);
  t = clock() - t;
  checkforward(m,n,Cd);
  printf("Forward mode with mxms: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);

  /* TEST 4: reverse mode with Tapenade */
  inittest(m,p,n,A,B);
  initreverse(m,n,Cb);
  t = clock();
  mxm_b(m,p,n,A,Ab,B,Bb,C,Cb);
  t = clock() - t;
  checkreverse(m,p,n,Ab,Bb);
  printf("Reverse mode with Tapenade: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);

  /* TEST 5: reverse mode with mxms */

  inittest(m,p,n,A,B);
  initreverse(m,n,Cb);
  t = clock();
  zeroout(m,p,Ab);
  zeroout(p,n,Bb);
  mxm_reverse(m,p,n,A,Ab,B,Bb,C,Cb);
  t = clock() - t;
  checkreverse(m,p,n,Ab,Bb);
  printf("Reverse mode with mxms: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);

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

void checkmult(int m, int n,double C[m][n])
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

void checkforward(int m,int n,double Cd[m][n])
{
  checkval(Cd[0][0],1.);checkval(Cd[0][1],1.);
  checkval(Cd[1][0],1.);checkval(Cd[1][1],1.);
}

void initreverse(int m,int n,double Cb[m][n])
{
  Cb[0][0] = 1.;Cb[0][1] = -1.;
  Cb[1][0] = -1.;Cb[1][1] = 1.;
}

void checkreverse(int m,int p,int n,double Ab[m][p],double Bb[p][n])
{
  checkval(Ab[0][0],1.);checkval(Ab[0][1],1.);checkval(Ab[0][2],1.);
  checkval(Ab[1][0],-1.);checkval(Ab[1][1],-1.);checkval(Ab[1][2],-1.);

  checkval(Bb[0][0],1.);checkval(Bb[0][1],-1.);
  checkval(Bb[1][0],1.);checkval(Bb[1][1],-1.);
  checkval(Bb[2][0],1.);checkval(Bb[2][1],-1.);
}
