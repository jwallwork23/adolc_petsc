#include "ex5_d.h"

/*
  Compute Jacobian using naive approach of propagating canonical Cartesian basis through the forward mode of AD.
*/
void ComputeJacobian(Field **f,Field **u,int gxs,int gxm,int gys,int gym,double hx,double hy,int My,void *ptr,double **J){

  int    i,j,l,k,n = 2*gxm*gym;
  Field  **fd,**seed;

  // TODO: Need to allocate memory for seed and fd

  for (j=gys; j<gym; j++) {
    for (i=gxs; i<gxm; i++) {
      seed[j][i].u = 1.;
      RHSLocal_d(f,fd,u,seed,gxs,gxm,gys,gym,hx,hy,ptr,ptr);
      seed[j][i].u = 0.;
      for (l=gys; l<gym; l++) {
        for (k=gxs; k<gxm; k++) {
          J[l][k] = fd[l][k].u;
        }
      }
      seed[j][i].v = 1.;
      RHSLocal_d(f,fd,u,seed,gxs,gxm,gys,gym,hx,hy,ptr,ptr);
      seed[j][i].v = 0.;
      for (l=gys; l<gym; l++) {
        for (k=gxs; k<gxm; k++) {
          J[l][k] = fd[l][k].v;
        }
      }
    }
  }

  // TODO: Need to free memory for seed and fd

}

