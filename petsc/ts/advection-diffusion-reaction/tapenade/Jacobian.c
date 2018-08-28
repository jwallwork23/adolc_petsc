#include "ex5_d.h"
#include "../utils.c"	// TODO: Replace all relative paths with absolute ones


void ComputeJacobian(Field **f,Field **u,int xs,int xm,int ys,int ym,double hx,double hy,int My,void *ptr,double J[2*(xs+xm)*(ys+ym)][2*(xs+xm)*(ys+ym)]){

  int    N = 2*(xs+xm)*(ys+ym),i,j,l,k;
  Field  **fd,**seed;

  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      seed[j][i].u = 1.;
      RHSLocal_d(f,fd,u,seed,xs,xm,ys,ym,hx,hy,ptr,ptr);
      seed[j][i].u = 0.;
      for (l=ys; l<ym; l++) {
        for (k=xs; k<xm; k++) {
          J[coord_map(k,l,0,My,2)][coord_map(i,j,0,My,2)] = fd[l][k].u;
        }
      }
      seed[j][i].v = 1.;
      RHSLocal_d(f,fd,u,seed,xs,xm,ys,ym,hx,hy,ptr,ptr);
      seed[j][i].v = 0.;
      for (l=ys; l<ym; l++) {
        for (k=xs; k<xm; k++) {
          J[coord_map(k,l,1,My,2)][coord_map(i,j,1,My,2)] = fd[l][k].v;
        }
      }
    }
  }
}
