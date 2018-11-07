#include <assert.h>
#include <math.h>
#include <time.h>

#include "derivatives/byhand.c"
#include "derivatives/dgemm_d.c"
#include "derivatives/dgemm_b.c"

// TODO: Use random matrices

void inittest(int m,int p,int n,double A[m][p],double B[p][n]);
void initforward(int m,int p,int n,double Ad[m][p],double Bd[p][n]);
void initreverse(int m,int n,double Cb[m][n]);

void checkval(double approx,double exact);
void checkforward(int m,int n,double Cd_mxm[m][n],double Cd_tapenade[m][n]);
void checkreverse(int m,int p,int n,double Ab_mxm[m][p],double Ab_tapenade[m][p],double Bb_mxm[p][n],double Bb_tapenade[p][n]);

int main(int argc,char* args[])
{
  clock_t t;
  int     m = 10,p = 10,n = 10,N = 1000,i;
  double  A[m][p],B[p][n],C[m][n],one = 1.,zero = 0.;
  double  Ad[m][p],Bd[p][n],Cd_byhand[m][n],Cd_tapenade[m][n];
  double  Ab_byhand[m][p],Ab_tapenade[m][p],Bb_byhand[p][n],Bb_tapenade[p][n],Cb[m][n];


  inittest(m,p,n,A,B);
  zeroout(m,n,C);
  t = clock();
  dgemm(0,0,m,A,B,C);
  t = clock() - t;
  printf("\n%30s: %.4e seconds","Single naive dgemm call",((double) t)/CLOCKS_PER_SEC);
  t = clock();
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,one,&A[0][0],m,&B[0][0],m,zero,&C[0][0],m);
  t = clock() - t;
  printf("\n%30s: %.4e seconds\n\n","Single lapack dgemm call",((double) t)/CLOCKS_PER_SEC);

  printf("*******************************************************************\n");
  printf("***** EXPERIMENT 1: differentiation w.r.t. both matrix inputs *****\n");
  printf("*******************************************************************\n\n");

  /* Forward mode with Tapenade */
  inittest(m,p,n,A,B);
  initforward(m,p,n,Ad,Bd);
  zeroout(m,n,C);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_d(m,p,n,A,Ad,B,Bd,C,Cd_tapenade);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Forward mode with Tapenade",((double) t)/CLOCKS_PER_SEC);

  /* Forward mode with dgemms */
  inittest(m,p,n,A,B);
  initforward(m,p,n,Ad,Bd);
  zeroout(m,n,C);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_dot(m,p,n,one,A,Ad,B,Bd,one,C,Cd_byhand);
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Forward mode with dgemms",((double) t)/CLOCKS_PER_SEC);
  checkforward(m,n,Cd_byhand,Cd_tapenade);

  /* Reverse mode with Tapenade */
  inittest(m,p,n,A,B);
  initreverse(m,n,Cb);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_b(m,p,n,A,Ab_tapenade,B,Bb_tapenade,C,Cb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Reverse mode with Tapenade",((double) t)/CLOCKS_PER_SEC);

  /* Reverse mode with dgemms */
  inittest(m,p,n,A,B);
  initreverse(m,n,Cb);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_bar(m,p,n,one,A,Ab_byhand,B,Bb_byhand,one,C,Cb);
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Reverse mode with dgemms",((double) t)/CLOCKS_PER_SEC);
  checkreverse(m,p,n,Ab_byhand,Ab_tapenade,Bb_byhand,Bb_tapenade);

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
