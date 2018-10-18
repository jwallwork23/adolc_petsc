#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>
#include "residuals_d.c"
//#include "residuals_dv.c"

//#include "ex46_d.h"  // TODO


/*
  (psi_i, u_j grad_j u_i) ==> (\psi_i, \phi_j grad_j u_i)

  TODO: Use Tapenade -multi option
*/
/*
static void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscScalar ud[dim][dim],u_td[dim][dim],f0[dim],fd0[dim][dim];
  PetscInt    i,j;

  for (i=0; i<dim; ++i) {
    for (j=0; j<dim; ++j) {
      if (i == j) {
      ud[i][j] = 1.;
      u_td[i][j] = u_tShift;
      } else {
      ud[i][j] = u_td[i][j] = 0.;
      }
    }
    f0[i] = 0.;
  }

  f0_mms1_u_dv(dim,Nf,NfAux,uOff,uOff_x,u,ud,u_t,u_td,u_x,aOff,aOff_x,a,a_t,a_x,t,x,numConstants,constants,f0,fd0,dim);

  for (i=0; i<dim; ++i) {
    for (j=0; j<dim; ++j) {
      g0[j*dim+i] = fd0[i][j];
    }
  }
}
*/


static void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscScalar ud[dim],u_td[dim],f0[dim],fd0[dim];
  PetscInt    i,j;

  for (i=0; i<dim; ++i) {
    ud[i] = 0.;
    u_td[i] = 0.;
    f0[i] = 0.;
  }

  for (i=0; i<dim; ++i) {
    ud[i] = 1.;
    u_td[i] = u_tShift;

    f0_mms1_u_d(dim,Nf,NfAux,uOff,uOff_x,u,ud,u_t,u_td,u_x,aOff,aOff_x,a,a_t,a_x,t,x,numConstants,constants,f0,fd0);
    for (j=0; j<dim; ++j)
      g0[j*dim+i] = fd0[j];

    ud[i] = 0.;
    u_td[i] = 0.;
  }
}

/* < q, \nabla\cdot u > 
   NcompI = 1, NcompJ = dim */
static void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscScalar u_xd[dim*dim],f0[dim],fd0[dim];
  PetscInt    i,j,k;

  for (i=0; i<dim; ++i) {
    u_xd[i] = 0.;
    f0[i] = 0.;
  }

  for (i=0; i<dim; ++i) {
    for (j=0; j<dim; ++j) {
      u_xd[j*dim+i] = 1.;
    
      f0_p_d(dim,Nf,NfAux,uOff,uOff_x,u,u_t,u_x,u_xd,aOff,aOff_x,a,a_t,a_x,t,x,numConstants,constants,f0,fd0);
      for (k=0; k<dim; ++k) {
        g1[(k*dim+j)*dim+i] = fd0[k];
      }

      u_xd[j*dim+i] = 0.;
    }
  }
}

/*
  (psi_i, u_j grad_j u_i) ==> (\psi_i, \u_j grad_j \phi_i)
*/
static void g1_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscScalar u_xd[dim*dim],f0[dim],fd0[dim];
  PetscInt    i,j,k;

  for (i=0; i<dim; ++i) {
    u_xd[i] = 0.;
    f0[i] = 0.;
  }

  for (i=0; i<dim; ++i) {
    for (j=0; j<dim; ++j) {
      u_xd[j*dim+i] = 1.;
    
      f0_mms2_u_d(dim,Nf,NfAux,uOff,uOff_x,u,u_t,u_x,u_xd,aOff,aOff_x,a,a_t,a_x,t,x,numConstants,constants,f0,fd0);
      for (k=0; k<dim; ++k) {
        g1[(k*dim+j)*dim+i] = fd0[k];
      }

      u_xd[j*dim+i] = 0.;
    }
  }
}

