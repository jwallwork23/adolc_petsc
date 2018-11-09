#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#include "derivatives/byhand.c"
#include "derivatives/naive_mtmv_d.c"
#include "derivatives/naive_mtmv_b.c"


void initrandom(int m,int n,double A[m][n]);

double getrand();
void checkval(double approx,double exact);
void checkvals(int m,int n,double C_byhand[m][n],double C_tapenade[m][n]);

int main(int argc,char* args[])
{
  clock_t t;
  int     m = 10,p = 10,n = 10,N = 1000,i; // TODO: Square everything off
  double  A[m][p],B[p][n],U[m][m],V_byhand[m][n],V_tapenade[m][n],one = 1.,zero = 0.,alpha,beta;
  double  Ad[m][p],Vd_byhand[m][n],Vd_tapenade[m][n];
  double  Ub_byhand[m][p],Ub_tapenade[m][p],Vb[m][n];

  printf("\n*******************************************************************\n");
  printf("***** EXPERIMENT 3: differentiation of tensor expression by U *****\n");
  printf("*****                                                         *****\n");
  printf("*****              Timings averaged over %4d runs            *****\n",N);
  printf("*******************************************************************\n");

  /* Assign (pseudo)random coefficients */
  alpha = getrand();
  beta = getrand();

  /* 'Naive' mtmv */
  initrandom(m,m,A);
  initrandom(m,m,B);
  initrandom(m,m,U);
  zeroout(m,m,V_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_mtmv(m,alpha,A,B,U,beta,V_tapenade);
  t = clock() - t;
  printf("\n%30s: %.4e seconds","Single naive mtmv call",((double) t)/(N*CLOCKS_PER_SEC));

  /* dgemm based mtmv */
  zeroout(m,m,V_byhand);
  t = clock();
  for (i=0; i<N; i++)
    mtmv(m,alpha,A,B,U,beta,V_byhand);
  t = clock() - t;
  printf("\n%30s: %.4e seconds\n\n","Single mtmv call using dgemm",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,m,V_byhand,V_tapenade);

// TODO below

  /* Forward mode with Tapenade */
/*
  initrandom(m,p,Ad);
  zeroout(m,n,C_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm_A_d(0,0,m,alpha,A,Ad,B,beta,C_tapenade,Cd_tapenade);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Forward mode with Tapenade",((double) t)/(N*CLOCKS_PER_SEC));
*/
  /* Forward mode with dgemms */
/*
  zeroout(m,n,C_byhand);
  t = clock();
  for (i=0; i<N; i++) {
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C_byhand[0][0],m);
    dgemm_A_dot(0,0,m,alpha,A,Ad,B,beta,C_byhand,Cd_byhand);
  }
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Forward mode with dgemms",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,n,Cd_byhand,Cd_tapenade);
*/
  /* Reverse mode with Tapenade */
/*
  initrandom(m,n,Cb);
  zeroout(m,n,Ab_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm_A_b(0,0,m,alpha,NULL,Ab_tapenade,B,beta,C_tapenade,Cb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Reverse mode with Tapenade",((double) t)/(N*CLOCKS_PER_SEC));
*/
  /* Reverse mode with dgemms */
/*
  zeroout(m,n,Ab_byhand);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_A_bar(0,0,m,alpha,NULL,Ab_byhand,B,beta,C_byhand,Cb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Reverse mode with dgemms",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,p,Ab_byhand,Ab_tapenade);
*/
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

void initrandom(int m,int n,double A[m][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      A[i][j] = getrand();
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
