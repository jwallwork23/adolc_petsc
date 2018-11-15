#include <time.h>
#include "utils.c"
#include "derivatives/byhand.c"
#include "derivatives/naive_dgemm_dA.c"
#include "derivatives/naive_dgemm_bA.c"


int main(int argc,char* args[])
{
  clock_t t;
  int     m = 10,N = 1000000,i;
  double  A[m][m],B[m][m],C_byhand[m][m],C_tapenade[m][m],alpha,beta;
  double  Ad[m][m],Cd_byhand[m][m],Cd_tapenade[m][m],time,time_dgemm;
  double  Ab_byhand[m][m],Ab_tapenade[m][m],Cb[m][m];

  printf("\n*******************************************************************\n");
  printf("***** EXPERIMENT 1: differentiation w.r.t. first matrix input *****\n");
  printf("*****                                                         *****\n");
  printf("*****            Timings averaged over %7d runs           *****\n",N);
  printf("*******************************************************************\n\n");

  /* Assign (pseudo)random coefficients and matrices */
  alpha = getrand();
  beta = getrand();
  initrandom(m,m,A);
  initrandom(m,m,B);
  initrandom(m,m,Ad);
  initrandom(m,m,Cb);

  /* Zero out output matrices */
  zeroout(m,m,C_tapenade);
  zeroout(m,m,C_byhand);
  zeroout(m,m,Ab_tapenade);
  zeroout(m,m,Ab_byhand);

  /* LAPACK dgemm */
  t = clock();
  for (i=0; i<N; i++)
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C_byhand[0][0],m);
  t = clock() - t;
  time_dgemm = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds\n","Single LAPACK dgemm call",time_dgemm);

  /* 'Naive' dgemm */
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm(0,0,m,alpha,A,B,beta,C_tapenade);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n\n","Single naive dgemm call",time,time/time_dgemm);
  checkvals(m,m,C_byhand,C_tapenade);

  /* Forward mode with Tapenade */
  zeroout(m,m,C_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm_dA(0,0,m,alpha,A,Ad,B,beta,C_tapenade,Cd_tapenade);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n","Forward mode with Tapenade",time,time/time_dgemm);

  /* Forward mode with dgemms */
  zeroout(m,m,C_byhand);
  t = clock();
  for (i=0; i<N; i++) {
    dgemm_A_dot(0,0,m,alpha,A,Ad,B,beta,C_byhand,Cd_byhand);
  }
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n\n","Forward mode with dgemms",time,time/time_dgemm);
  checkvals(m,m,Cd_byhand,Cd_tapenade);

  /* Reverse mode with Tapenade */
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm_bA(0,0,m,alpha,NULL,Ab_tapenade,B,beta,C_tapenade,Cb);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n","Reverse mode with Tapenade",time,time/time_dgemm);

  /* Reverse mode with dgemms */
  t = clock();
  for (i=0; i<N; i++)
    dgemm_A_bar(0,0,m,alpha,NULL,Ab_byhand,B,beta,C_byhand,Cb);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n\n","Reverse mode with dgemms",time,time/time_dgemm);
  checkvals(m,m,Ab_byhand,Ab_tapenade);

  printf("All tests passed.\n");

  return 0;
}
