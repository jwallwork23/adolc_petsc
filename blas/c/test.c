#include <assert.h>
#include <math.h>
#include <time.h>

#include "derivatives/byhand.c"
#include "derivatives/dgemm_d.c"
#include "derivatives/dgemm_b.c"


void inittest(int m,int p,int n,double A[m][p],double B[p][n]);
void initforward(int m,int p,int n,double Ad[m][p],double Bd[p][n]);
void initreverse(int m,int n,double Cb[m][n]);

void checkval(double approx,double exact);
void checkforward(int m,int n,double Cd_mxm[m][n],double Cd_tapenade[m][n]);
void checkreverse(int m,int p,int n,double Ab_mxm[m][p],double Ab_tapenade[m][p],double Bb_mxm[p][n],double Bb_tapenade[p][n]);

int main(int argc,char* args[])
{
  clock_t t;
  int     m,p,n;

  //printf("m ?= ");
  //scanf("%d",&m);
  m = 250;
  p = m;n = m;

  double  A[m][p],B[p][n],C[m][n];
  double  Ad[m][p],Bd[p][n],Cd_mxm[m][n],Cd_tapenade[m][n];
  double  Ab_mxm[m][p],Ab_tapenade[m][p],Bb_mxm[p][n],Bb_tapenade[p][n],Cb[m][n];


  /* TEST 1: matrix-matrix multiply */

  inittest(m,p,n,A,B);
  zeroout(m,n,C);
  t = clock();
  dgemm(0,0,m,A,B,C);
  t = clock() - t;
  printf("Original mxm call: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);

  /* TEST 2: forward mode with Tapenade */

  inittest(m,p,n,A,B);
  initforward(m,p,n,Ad,Bd);
  zeroout(m,n,C);
  t = clock();
  dgemm_d(m,p,n,A,Ad,B,Bd,C,Cd_tapenade);
  t = clock() - t;
  printf("Forward mode with Tapenade: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);

  /* TEST 3: forward mode with mxms */

  inittest(m,p,n,A,B);
  initforward(m,p,n,Ad,Bd);
  zeroout(m,n,C);
  t = clock();
  zeroout(m,n,Cd_mxm);
  dgemm_dot(m,p,n,A,Ad,B,Bd,C,Cd_mxm);
  t = clock() - t;
  printf("Forward mode with dgemms: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);
  checkforward(m,n,Cd_mxm,Cd_tapenade);

  /* TEST 4: reverse mode with Tapenade */
  inittest(m,p,n,A,B);
  initreverse(m,n,Cb);
  t = clock();
  dgemm_b(m,p,n,A,Ab_tapenade,B,Bb_tapenade,C,Cb);
  t = clock() - t;
  printf("Reverse mode with Tapenade: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);

  /* TEST 5: reverse mode with mxms */

  inittest(m,p,n,A,B);
  initreverse(m,n,Cb);
  t = clock();
  zeroout(m,p,Ab_mxm);
  zeroout(p,n,Bb_mxm);
  dgemm_bar(m,p,n,A,Ab_mxm,B,Bb_mxm,C,Cb);
  t = clock() - t;
  printf("Reverse mode with dgemms: %.4e seconds\n",((double) t)/CLOCKS_PER_SEC);
  checkreverse(m,p,n,Ab_mxm,Ab_tapenade,Bb_mxm,Bb_tapenade);

  printf("All tests passed.\n");

  return 0;
}

void checkval(double approx,double exact)
{
  assert(fabs(approx-exact)<1e-8);
}

void inittest(int m,int p,int n,double A[m][p],double B[p][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      A[i][j] = i+j;
    }
  }

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      B[i][j] = -i;
    }
  }
}

void initforward(int m,int p,int n,double Ad[m][p],double Bd[p][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      Ad[i][j] = i-j*j;
    }
  }

  for (i=0; i<p; i++) {
    for (j=0; j<n; j++) {
      Bd[i][j] = 3*j;
    }
  }
}

void checkforward(int m,int n,double Cd_mxm[m][n],double Cd_tapenade[m][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      checkval(Cd_mxm[i][j],Cd_tapenade[i][j]);
    }
  }
}

void initreverse(int m,int n,double Cb[m][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      Cb[i][j] = i*i;
    }
  }
}

void checkreverse(int m,int p,int n,double Ab_mxm[m][p],double Ab_tapenade[m][p],double Bb_mxm[p][n],double Bb_tapenade[p][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      checkval(Ab_mxm[i][j],Ab_tapenade[i][j]);
    }
  }
  for (i=0; i<p; i++) {
    for (j=0; j<n; j++) {
      checkval(Bb_mxm[i][j],Bb_tapenade[i][j]);
    }
  }
}
