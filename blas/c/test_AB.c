#include <time.h>
#include "utils.c"
#include "derivatives/byhand.c"
#include "derivatives/naive_dgemm_d.c"
#include "derivatives/naive_dgemm_b.c"


int main(int argc,char* args[])
{
  clock_t t;
  int     m = 10,N = 1000000,i;
  double  A[m][m],B[m][m],C_byhand[m][m],C_tapenade[m][m],one = 1.,zero = 0.,alpha,beta;
  double  Ad[m][m],Bd[m][m],Cd_byhand[m][m],Cd_tapenade[m][m],time,time_dgemm;
  double  Ab_byhand[m][m],Ab_tapenade[m][m],Bb_byhand[m][m],Bb_tapenade[m][m],Cb[m][m];

  printf("\n*******************************************************************\n");
  printf("***** EXPERIMENT 2: differentiation w.r.t. both matrix inputs *****\n");
  printf("*****                                                         *****\n");
  printf("*****            Timings averaged over %7d runs           *****\n",N);
  printf("*******************************************************************\n\n");

  /* Assign (psuedo)random coefficients and matrices*/
  alpha = getrand();
  beta = getrand();
  initrandom(m,m,A);
  initrandom(m,m,B);
  initrandom(m,m,Ad);
  initrandom(m,m,Bd);
  initrandom(m,m,Cb);

  /* Zero out output matrices */
  zeroout(m,m,C_tapenade);
  zeroout(m,m,C_byhand);
  zeroout(m,m,Ab_tapenade);
  zeroout(m,m,Bb_tapenade);
  zeroout(m,m,Ab_byhand);
  zeroout(m,m,Bb_byhand);

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
    naive_dgemm_d(0,0,m,alpha,A,Ad,B,Bd,beta,C_tapenade,Cd_tapenade);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n","Forward mode with Tapenade",time,time/time_dgemm);

  /* Forward mode with dgemms */
  zeroout(m,m,C_byhand);
  t = clock();
  for (i=0; i<N; i++) {
    dgemm_dot(0,0,m,alpha,A,Ad,B,Bd,one,C_byhand,Cd_byhand);
  }
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n\n","Forward mode with dgemms",time,time/time_dgemm);
  checkvals(m,m,Cd_byhand,Cd_tapenade);

  /* Reverse mode with Tapenade */
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm_b(0,0,m,alpha,A,Ab_tapenade,B,Bb_tapenade,beta,C_tapenade,Cb);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n","Reverse mode with Tapenade",time,time/time_dgemm);

  /* Reverse mode with dgemms */
  t = clock();
  for (i=0; i<N; i++)
    dgemm_bar(0,0,m,alpha,A,Ab_byhand,B,Bb_byhand,beta,C_byhand,Cb);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n\n","Reverse mode with dgemms",time,time/time_dgemm);
  checkvals(m,m,Ab_byhand,Ab_tapenade);
  checkvals(m,m,Bb_byhand,Bb_tapenade);

  printf("All tests passed.\n");

  return 0;
}
