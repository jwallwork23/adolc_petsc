#include <assert.h>
#include <math.h>
#include <time.h>

#include "derivatives/byhand.c"
#include "derivatives/dgemm_A_d.c"
#include "derivatives/dgemm_A_b.c"

// TODO: Use random matrices
//#include <stdlib.h>
//int r = rand();
// Possibly use modulo r % 5 or something

void inittest(int m,int p,int n,double A[m][p],double B[p][n]);
void initforward(int m,int p,int n,double Ad[m][p]);
void initreverse(int m,int n,double Cb[m][n]);

void checkval(double approx,double exact);
void checkmxm(int m,int n,double C_mxm[m][n],double C_tapenade[m][n]);
void checkforward(int m,int n,double Cd_mxm[m][n],double Cd_tapenade[m][n]);
void checkreverse(int m,int p,int n,double Ab_mxm[m][p],double Ab_tapenade[m][p]);

int main(int argc,char* args[])
{
  clock_t t;
  int     m = 10,p = 10,n = 10,N = 1000,i;
  double  A[m][p],B[p][n],C_byhand[m][n],C_tapenade[m][n],one = 1.,zero = 0.,alpha = 0.9,beta=1.1;
  double  Ad[m][p],Cd_byhand[m][n],Cd_tapenade[m][n];
  double  Ab_byhand[m][p],Ab_tapenade[m][p],Cb[m][n];


  printf("\n*******************************************************************\n");
  printf("***** EXPERIMENT 2: differentiation w.r.t. first matrix input *****\n");
  printf("*******************************************************************\n");

  inittest(m,p,n,A,B);
  zeroout(m,n,C_tapenade);
  zeroout(m,n,C_byhand);
  t = clock();
  dgemm(0,0,m,alpha,A,B,beta,C_byhand);
  t = clock() - t;
  printf("\n%30s: %.4e seconds","Single naive dgemm call",((double) t)/CLOCKS_PER_SEC);
  t = clock();
  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C_tapenade[0][0],m);
  t = clock() - t;
  printf("\n%30s: %.4e seconds\n\n","Single lapack dgemm call",((double) t)/CLOCKS_PER_SEC);
  checkmxm(m,n,C_byhand,C_tapenade);

  /* Forward mode with Tapenade */
  inittest(m,p,n,A,B);
  initforward(m,p,n,Ad);
  zeroout(m,n,C_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_A_d(0,0,m,alpha,A,Ad,B,beta,C_tapenade,Cd_tapenade);
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Forward mode with Tapenade",((double) t)/CLOCKS_PER_SEC);

  /* Forward mode with dgemms */
  inittest(m,p,n,A,B);
  initforward(m,p,n,Ad);
  zeroout(m,n,C_byhand);
  t = clock();
  for (i=0; i<N; i++) {
    cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,m,m,alpha,&A[0][0],m,&B[0][0],m,beta,&C_byhand[0][0],m);
    dgemm_A_dot(0,0,m,alpha,A,Ad,B,beta,C_byhand,Cd_byhand);
  }
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Forward mode with dgemms",((double) t)/CLOCKS_PER_SEC);
  checkforward(m,n,Cd_byhand,Cd_tapenade);

  /* Reverse mode with Tapenade */
  inittest(m,p,n,A,B);
  initreverse(m,n,Cb);
  zeroout(m,n,Ab_tapenade);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_A_b(0,0,m,alpha,A,Ab_tapenade,B,one,C_tapenade,Cb);	// FIXME: beta
  t = clock() - t;
  printf("%30s: %.4e seconds\n","Reverse mode with Tapenade",((double) t)/CLOCKS_PER_SEC);

  /* Reverse mode with dgemms */
  inittest(m,p,n,A,B);
  initreverse(m,n,Cb);
  zeroout(m,n,Ab_byhand);
  t = clock();
  for (i=0; i<N; i++)
    dgemm_A_bar(0,0,m,alpha,A,Ab_byhand,B,one,C_byhand,Cb);	// FIXME: beta
  t = clock() - t;
  printf("%30s: %.4e seconds\n\n","Reverse mode with dgemms",((double) t)/CLOCKS_PER_SEC);
  checkreverse(m,p,n,Ab_byhand,Ab_tapenade);

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

void initforward(int m,int p,int n,double Ad[m][p])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      Ad[i][j] = i-j*j;
    }
  }
}

void checkmxm(int m,int n,double C_mxm[m][n],double C_tapenade[m][n])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      checkval(C_mxm[i][j],C_tapenade[i][j]);
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

void checkreverse(int m,int p,int n,double Ab_mxm[m][p],double Ab_tapenade[m][p])
{
  int i,j;

  for (i=0; i<m; i++) {
    for (j=0; j<p; j++) {
      checkval(Ab_mxm[i][j],Ab_tapenade[i][j]);
    }
  }
}
