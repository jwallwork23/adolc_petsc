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
  int     m = 10,N = 1000,i;
  double  A[m][m],B[m][m],U[m][m],V_byhand[m][m],V_tapenade[m][m],one = 1.,zero = 0.,alpha,beta;
  double  Ud[m][m],Vd_byhand[m][m],Vd_tapenade[m][m];
  double  Ub_byhand[m][m],Ub_tapenade[m][m],Vb[m][m];

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

  /* Forward mode with Tapenade */
  initrandom(m,m,Ud);
  zeroout(m,m,V_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_mtmv_d(m,alpha,A,B,U,Ud,beta,V_tapenade,Vd_tapenade);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Forward mode with Tapenade",((double) t)/(N*CLOCKS_PER_SEC));

  /* Forward mode with dgemms */
  zeroout(m,m,V_byhand);
  t = clock();
  for (i=0; i<N; i++) {
    mtmv_dot(m,alpha,A,B,U,Ud,beta,V_byhand,Vd_byhand);
  }
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Forward mode with dgemms",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,m,Vd_byhand,Vd_tapenade);

  /* Reverse mode with Tapenade */
  initrandom(m,m,Vb);
  zeroout(m,m,Ub_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_mtmv_b(m,alpha,A,B,U,Ub_tapenade,beta,V_tapenade,Vb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Reverse mode with Tapenade",((double) t)/(N*CLOCKS_PER_SEC));

  /* Reverse mode with dgemms */
  zeroout(m,m,Ub_byhand);
  t = clock();
  for (i=0; i<N; i++)
    mtmv_bar(m,alpha,A,B,U,Ub_byhand,beta,V_byhand,Vb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Reverse mode with dgemms",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,m,Ub_byhand,Ub_tapenade);

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
