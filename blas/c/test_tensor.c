#include <time.h>
#include "utils.c"
#include "derivatives/byhand.c"
#include "derivatives/naive_mtmv_d.c"
#include "derivatives/naive_mtmv_b.c"


int main(int argc,char* args[])
{
  clock_t t;
  int     m = 10,N = 1000000,i;
  double  A[m][m],B[m][m],U[m][m],V_byhand[m][m],V_tapenade[m][m],one = 1.,zero = 0.,alpha,beta;
  double  Ud[m][m],Vd_byhand[m][m],Vd_tapenade[m][m],time,time_dgemm;
  double  Ub_byhand[m][m],Ub_tapenade[m][m],Vb[m][m];

  printf("\n*******************************************************************\n");
  printf("***** EXPERIMENT 3: differentiation of tensor expression by U *****\n");
  printf("*****                                                         *****\n");
  printf("*****            Timings averaged over %7d runs           *****\n",N);
  printf("*******************************************************************\n\n");

  /* Assign (pseudo)random coefficients and matrices */
  alpha = getrand();
  beta = getrand();
  initrandom(m,m,A);
  initrandom(m,m,B);
  initrandom(m,m,U);
  initrandom(m,m,Ud);
  initrandom(m,m,Vb);

  /* Zero out output matrices */
  zeroout(m,m,V_byhand);
  zeroout(m,m,V_tapenade);
  zeroout(m,m,Ub_tapenade);
  zeroout(m,m,Ub_byhand);

  /* LAPACK dgemm */
  t = clock();
  for (i=0; i<N; i++)
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&V_byhand[0][0],m);
  t = clock() - t;
  time_dgemm = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds\n\n","Single LAPACK dgemm call",time_dgemm);

  /* dgemm based mtmv */
  zeroout(m,m,V_byhand);
  t = clock();
  for (i=0; i<N; i++)
    mtmv(m,alpha,A,B,U,beta,V_byhand);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n","Single mtmv call using dgemm",time,time/time_dgemm);

  /* 'Naive' mtmv */
  t = clock();
  for (i=0; i<N; i++)
    naive_mtmv(m,alpha,A,B,U,beta,V_tapenade);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n\n","Single naive mtmv call",time,time/time_dgemm);
  checkvals(m,m,V_byhand,V_tapenade);

  /* Forward mode with Tapenade */
  zeroout(m,m,V_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    naive_mtmv_d(m,alpha,A,B,U,Ud,beta,V_tapenade,Vd_tapenade);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n","Forward mode with Tapenade",time,time/time_dgemm);

  /* Forward mode with dgemms */
  zeroout(m,m,V_byhand);
  t = clock();
  for (i=0; i<N; i++) {
    mtmv_dot(m,alpha,A,B,U,Ud,beta,V_byhand,Vd_byhand);
  }
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n\n","Forward mode with dgemms",time,time/time_dgemm);
  checkvals(m,m,Vd_byhand,Vd_tapenade);

  /* Reverse mode with Tapenade */
  t = clock();
  for (i=0; i<N; i++)
    naive_mtmv_b(m,alpha,A,B,U,Ub_tapenade,beta,V_tapenade,Vb);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n","Reverse mode with Tapenade",time,time/time_dgemm);

  /* Reverse mode with dgemms */
  t = clock();
  for (i=0; i<N; i++)
    mtmv_bar(m,alpha,A,B,U,Ub_byhand,beta,V_byhand,Vb);
  t = clock() - t;
  time = ((double) t)/(N*CLOCKS_PER_SEC);
  printf("%30s: %.4e seconds (%6.4f dgemms)\n\n","Reverse mode with dgemms",time,time/time_dgemm);
  checkvals(m,m,Ub_byhand,Ub_tapenade);

  printf("All tests passed.\n");

  return 0;
}
