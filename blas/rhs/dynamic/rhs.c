#include <stdlib.h>
#include "mxm.h"

void RHS(int m,Field **ul,double **mass,double **stiff,double **grad,double Lex,double Ley,double mu,int ix,int iy,Field **outl)
{
  double alpha,beta,one = 1.;
  int    i,jx,jy,indx,indy;

  double **ulb = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) ulb[i] = malloc(m*sizeof(double));

  double **vlb = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) vlb[i] = malloc(m*sizeof(double));

  double **wrk2 = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) wrk2[i] = malloc(m*sizeof(double));

  double **wrk3 = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) wrk3[i] = malloc(m*sizeof(double));

  double **wrk4 = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) wrk4[i] = malloc(m*sizeof(double));

  double **wrk5 = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) wrk5[i] = malloc(m*sizeof(double));

  double **wrk6 = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) wrk6[i] = malloc(m*sizeof(double));

  double **wrk7 = malloc(m*sizeof(double*));
  for (i=0; i<m; i++) wrk7[i] = malloc(m*sizeof(double));

  for (jx=0; jx<m; jx++) {
    for (jy=0; jy<m; jy++) {
      ulb[jy][jx] = 0.;
      vlb[jy][jx] = 0.;
      indx = ix*(m-1)+jx;
      indy = iy*(m-1)+jy;
      ulb[jy][jx] = ul[indy][indx].u;
      vlb[jy][jx] = ul[indy][indx].v;
    }
  }

  alpha = Lex/Ley;
  extra_naive_mtmv(m,alpha,stiff,mass,ulb,beta,wrk2);

  alpha = Lex/Ley;
  extra_naive_mtmv(m,alpha,mass,stiff,ulb,beta,wrk3);

  naive_matrix_axpy(m,one,wrk3,wrk2);

  alpha = Lex/Ley;
  extra_naive_mtmv(m,alpha,stiff,mass,vlb,beta,wrk3);

  alpha = Lex/Ley;
  extra_naive_mtmv(m,alpha,mass,stiff,vlb,beta,wrk4);

  naive_matrix_axpy(m,one,wrk4,wrk3);

  alpha = Lex/2.;
  extra_naive_mtmv(m,alpha,grad,mass,ulb,beta,wrk4);

  naive_mpm(m,m,wrk4,ulb,wrk4);
  alpha = Ley/2.;
  extra_naive_mtmv(m,alpha,mass,grad,ulb,beta,wrk5);
  naive_mpm(m,m,wrk5,vlb,wrk5);

  naive_matrix_axpy(m,one,wrk5,wrk4);

  alpha = Lex/2.;
  extra_naive_mtmv(m,alpha,grad,mass,vlb,beta,wrk6);
  naive_mpm(m,m,wrk6,ulb,wrk6);

  alpha = Ley/2.;
  extra_naive_mtmv(m,alpha,mass,grad,vlb,beta,wrk7);
  naive_mpm(m,m,wrk7,vlb,wrk7);

  naive_matrix_axpy(m,one,wrk7,wrk6);

  for (jx=0; jx<m; jx++) {
    for (jy=0; jy<m; jy++) {
      indx = ix*(m-1)+jx;
      indy = iy*(m-1)+jy;
      outl[indy][indx].u += mu*wrk2[jy][jx]+wrk4[jy][jx];
      outl[indy][indx].v += mu*wrk3[jy][jx]+wrk6[jy][jx];
    }
  }

  for (i=0; i<m; i++) free(wrk7[i]);
  free(wrk7);

  for (i=0; i<m; i++) free(wrk6[i]);
  free(wrk6);

  for (i=0; i<m; i++) free(wrk5[i]);
  free(wrk5);

  for (i=0; i<m; i++) free(wrk4[i]);
  free(wrk4);

  for (i=0; i<m; i++) free(wrk3[i]);
  free(wrk3);

  for (i=0; i<m; i++) free(wrk2[i]);
  free(wrk2);

  for (i=0; i<m; i++) free(vlb[i]);
  free(vlb);

  for (i=0; i<m; i++) free(ulb[i]);
  free(ulb);
}
