#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include "derivatives/byhand.c"
#include "derivatives/naive_dgemm_d.c"
#include "derivatives/naive_dgemm_b.c"


void inittest(int m,int p,int n,double A[m][p],double B[p][n]);
void initforward(int m,int p,int n,double Ad[m][p],double Bd[p][n]);
void initreverse(int m,int n,double Cb[m][n]);

double getrand();
void checkval(double approx,double exact);
void checkvals(int m,int n,double C_byhand[m][n],double C_tapenade[m][n]);

int main(int argc,char* args[])
{
  clock_t t;
  int     m = 10,p = 10,n = 10,N = 1000,i; // TODO: Square everything off
  double  A[m][p],B[p][n],C_byhand[m][n],C_tapenade[m][n],one = 1.,zero = 0.,alpha,beta;
  double  Ad[m][p],Bd[p][n],Cd_byhand[m][n],Cd_tapenade[m][n];
  double  Ab_byhand[m][p],Ab_tapenade[m][p],Bb_byhand[p][n],Bb_tapenade[p][n],Cb[m][n];

  printf("\n*******************************************************************\n");
  printf("***** EXPERIMENT 2: differentiation w.r.t. both matrix inputs *****\n");
  printf("*****                                                         *****\n");
  printf("*****              Timings averaged over %4d runs            *****\n",N);
  printf("*******************************************************************\n");

  /* Assign (psuedo)random coefficients */
  alpha = getrand();
  beta = getrand();

  /* 'Naive' dgemm */
  inittest(m,p,n,A,B);
  zeroout(m,n,C_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm(0,0,m,alpha,A,B,beta,C_tapenade);
  t = clock() - t;
  printf("\n%30s: %.4e seconds","Single naive dgemm call",((double) t)/(N*CLOCKS_PER_SEC));

  /* LAPACK dgemm */
  zeroout(m,n,C_byhand);
  t = clock();
  for (i=0; i<N; i++)
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C_byhand[0][0],m);
  t = clock() - t;
  printf("\n%30s: %.4e seconds\n\n","Single LAPACK dgemm call",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,n,C_byhand,C_tapenade);

  /* Forward mode with Tapenade */
  initforward(m,p,n,Ad,Bd);
  zeroout(m,n,C_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm_d(0,0,m,alpha,A,Ad,B,Bd,beta,C_tapenade,Cd_tapenade);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Forward mode with Tapenade",((double) t)/(N*CLOCKS_PER_SEC));

  /* Forward mode with dgemms */
  zeroout(m,n,C_byhand);
  t = clock();
  for (i=0; i<N; i++) {
    dgemm_dot(0,0,m,alpha,A,Ad,B,Bd,one,C_byhand,Cd_byhand);
  }
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Forward mode with dgemms",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,n,Cd_byhand,Cd_tapenade);

  /* Reverse mode with Tapenade */
  initreverse(m,n,Cb);
  zeroout(m,n,Ab_tapenade);
  zeroout(m,n,Bb_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm_b(0,0,m,alpha,A,Ab_tapenade,B,Bb_tapenade,beta,C_tapenade,Cb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Reverse mode with Tapenade",((double) t)/(N*CLOCKS_PER_SEC));

  /* Reverse mode with dgemms */
  zeroout(m,n,Ab_byhand);
  zeroout(m,n,Bb_byhand);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_bar(0,0,m,alpha,A,Ab_byhand,B,Bb_byhand,beta,C_byhand,Cb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Reverse mode with dgemms",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,p,Ab_byhand,Ab_tapenade);
  checkvals(p,n,Bb_byhand,Bb_tapenade);

  printf("All tests passed.\n");

  return 0;
}

void checkval(double approx,double exact)
{
  assert(fabs(approx-exact)<1e-8);
}

double getrand()
{
  return ((double) rand()) / 1.e9 - 1.;
}

void inittest(int m,int p,int n,double A[m][p],double B[p][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      A[i][j] = getrand();
    }
  }

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      B[i][j] = getrand();
    }
  }
}

void initforward(int m,int p,int n,double Ad[m][p],double Bd[p][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      Ad[i][j] = getrand();
    }
  }

  for (i=0; i<p; i++) {
    for (j=0; j<n; j++) {
      Bd[i][j] = getrand();
    }
  }
}

void initreverse(int m,int n,double Cb[m][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      Cb[i][j] = getrand();
    }
  }
}

void checkvals(int m,int n,double C_byhand[m][n],double C_tapenade[m][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      checkval(C_byhand[i][j],C_tapenade[i][j]);
    }
  }
}
