#include "mxm.h"

typedef struct {
  double u,v;
} Field;

void RHS(int m,Field ul[m][m],double mass[m][m],double stiff[m][m],double grad[m][m],double Lex,double Ley,double mu,int ix,int iy,Field outl[m][m])
{
  double ulb[m][m],vlb[m][m],wrk2[m][m],wrk3[m][m],wrk4[m][m],wrk5[m][m],wrk6[m][m],wrk7[m][m];
  double alpha,beta,one = 1.;
  int    jx,jy,indx,indy;

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
}
