#include <petscdm.h>
#include <petscdmda.h>
#include <adolc/adolc.h>

extern PetscErrorCode GiveGhostPoints2d(DM da,void *cgs,void *a2d);
extern PetscErrorCode AdoubleGiveGhostPoints2d(DM da,adouble *cgs,adouble **a2d[]);
extern PetscErrorCode ConvertTo1Array(DM da,PetscScalar **u,PetscScalar *u_vec);
extern PetscErrorCode ConvertTo1Array2d(DM da,PetscScalar **u,PetscScalar *u_vec);
extern PetscErrorCode Subidentity(PetscInt p,PetscInt s,PetscScalar **S);

/*@C
  Wrapper function for allocating contiguous memory in a 2d array

  Input parameters:
  m,n - number of rows and columns of array, respectively

  Outpu parameter:
  A   - pointer to array for which memory is allocated
@*/
PetscErrorCode AdolcMalloc2(PetscInt m,PetscInt n,PetscScalar **A[])
{
  PetscFunctionBegin;
  *A = myalloc2(m,n);
  PetscFunctionReturn(0);
}

/*@C
  Wrapper function for freeing memory allocated with AdolcMalloc2

  Input parameter:
  A - array to free memory of
@*/
PetscErrorCode AdolcFree2(PetscScalar **A)
{
  PetscFunctionBegin;
  myfree2(A);
  PetscFunctionReturn(0);
}

/*@C
  TODO: Documentation
@*/
PetscErrorCode GiveGhostPoints2d(DM da,void *cgs,void *a2d)
{

  PetscFunctionBegin;

  // TODO general format accounting for dimensions and dofs

  PetscFunctionReturn(0);
}

/*@C
  Shift indices in adouble array to endow it with ghost points.

  Input parameters:
  da  - distributed array upon which variables are defined
  cgs - contiguously allocated 1-array with as many entries as there are
        interior and ghost points, in total

  Output parameter:
  a2d - contiguously allocated 2-array with ghost points, pointing to the
        1-array
@*/
PetscErrorCode AdoubleGiveGhostPoints2d(DM da,adouble *cgs,adouble **a2d[])
{
  PetscErrorCode ierr;
  PetscInt       gxs,gys,gxm,gym,j;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  for (j=0; j<gym; j++)
    (*a2d)[j] = cgs + j*gxm - gxs;
  *a2d -= gys;
  PetscFunctionReturn(0);
}

/*@C
  TODO: Documentation
  FIXME and generalise for dimensions and dofs
@*/
PetscErrorCode ConvertTo1Array(DM da,void **u,PetscScalar *u_vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ConvertTo1Array2d(da,(PetscScalar**)u,u_vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  Convert a 2-array defined on a DMDA to a 1-array

  Input parameters:
  da    - distributed array upon which variables are defined
  u     - 2-array to be converted

  Output parameters:
  u_vec - corresponding 1-array

  TODO: Generalise along with ConvertTo1Array above
@*/
PetscErrorCode ConvertTo1Array2d(DM da,PetscScalar **u,PetscScalar *u_vec)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k = 0,gxs,gys,gxm,gym;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++)
      u_vec[k++] = u[j][i];
  }
  PetscFunctionReturn(0);
}

/* TODO: docs */
PetscErrorCode Subidentity(PetscInt p,PetscInt s,PetscScalar **S)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<p; i++) {
    S[i+s][i] = 1.;
  }
  PetscFunctionReturn(0);
}
