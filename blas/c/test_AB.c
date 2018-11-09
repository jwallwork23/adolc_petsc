#include <time.h>
#include "utils.c"
#include "derivatives/byhand.c"
#include "derivatives/naive_dgemm_d.c"
#include "derivatives/naive_dgemm_b.c"


int main(int argc,char* args[])
{
  clock_t t;
  int     m = 10,N = 1000,i;
  double  A[m][m],B[m][m],C_byhand[m][m],C_tapenade[m][m],one = 1.,zero = 0.,alpha,beta;
  double  Ad[m][m],Bd[m][m],Cd_byhand[m][m],Cd_tapenade[m][m];
  double  Ab_byhand[m][m],Ab_tapenade[m][m],Bb_byhand[m][m],Bb_tapenade[m][m],Cb[m][m];

  printf("\n*******************************************************************\n");
  printf("***** EXPERIMENT 2: differentiation w.r.t. both matrix inputs *****\n");
  printf("*****                                                         *****\n");
  printf("*****              Timings averaged over %4d runs            *****\n",N);
  printf("*******************************************************************\n");

  /* Assign (psuedo)random coefficients */
  alpha = getrand();
  beta = getrand();

  /* 'Naive' dgemm */
  initrandom(m,m,A);
  initrandom(m,m,B);
  zeroout(m,m,C_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm(0,0,m,alpha,A,B,beta,C_tapenade);
  t = clock() - t;
  printf("\n%30s: %.4e seconds","Single naive dgemm call",((double) t)/(N*CLOCKS_PER_SEC));

  /* LAPACK dgemm */
  zeroout(m,m,C_byhand);
  t = clock();
  for (i=0; i<N; i++)
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C_byhand[0][0],m);
  t = clock() - t;
  printf("\n%30s: %.4e seconds\n\n","Single LAPACK dgemm call",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,m,C_byhand,C_tapenade);

  /* Forward mode with Tapenade */
  initrandom(m,m,Ad);
  initrandom(m,m,Bd);
  zeroout(m,m,C_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm_d(0,0,m,alpha,A,Ad,B,Bd,beta,C_tapenade,Cd_tapenade);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Forward mode with Tapenade",((double) t)/(N*CLOCKS_PER_SEC));

  /* Forward mode with dgemms */
  zeroout(m,m,C_byhand);
  t = clock();
  for (i=0; i<N; i++) {
    dgemm_dot(0,0,m,alpha,A,Ad,B,Bd,one,C_byhand,Cd_byhand);
  }
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Forward mode with dgemms",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,m,Cd_byhand,Cd_tapenade);

  /* Reverse mode with Tapenade */
  initrandom(m,m,Cb);
  zeroout(m,m,Ab_tapenade);
  zeroout(m,m,Bb_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_dgemm_b(0,0,m,alpha,A,Ab_tapenade,B,Bb_tapenade,beta,C_tapenade,Cb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Reverse mode with Tapenade",((double) t)/(N*CLOCKS_PER_SEC));

  /* Reverse mode with dgemms */
  zeroout(m,m,Ab_byhand);
  zeroout(m,m,Bb_byhand);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_bar(0,0,m,alpha,A,Ab_byhand,B,Bb_byhand,beta,C_byhand,Cb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Reverse mode with dgemms",((double) t)/(N*CLOCKS_PER_SEC));
  checkvals(m,m,Ab_byhand,Ab_tapenade);
  checkvals(m,m,Bb_byhand,Bb_tapenade);

  printf("All tests passed.\n");

  return 0;
}
