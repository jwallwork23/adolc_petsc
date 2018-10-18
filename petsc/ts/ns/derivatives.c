#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>
#include "residuals_d.c"


/*
  (psi_i, u_j grad_j u_i) ==> (\psi_i, \phi_j grad_j u_i)

  TODO: Use Tapenade -multi option
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
    for (j=0; j<dim; ++j) {
      g0[j*dim+i] = fd0[j];
    }

    ud[i] = 0.;
    u_td[i] = 0.;
  }
}

