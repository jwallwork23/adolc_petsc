#include <petscdm.h>
#include <petscdmda.h>
#include <adolc/adolc.h>


/*
  Wrapper function for allocating contiguous memory in a 2d array

  Input parameters:
  m,n - number of rows and columns of array, respectively

  Outpu parameter:
  A   - pointer to array for which memory is allocated

  TODO: Account for integer arrays
*/
template <class T>
PetscErrorCode AdolcMalloc2(PetscInt m,PetscInt n,T **A[])
{
  PetscFunctionBegin;
  *A = myalloc2(m,n);
  PetscFunctionReturn(0);
}

/*
  Wrapper function for freeing memory allocated with AdolcMalloc2

  Input parameter:
  A - array to free memory of

  TODO: Account for integer arrays
*/
template <class T>
PetscErrorCode AdolcFree2(T **A)
{
  PetscFunctionBegin;
  myfree2(A);
  PetscFunctionReturn(0);
}

/*
  Shift indices in an array of type T to endow it with ghost points.
  (e.g. This works for arrays of adoubles or AFields.)

  Input parameters:
  da   - distributed array upon which variables are defined
  cgs  - contiguously allocated 1-array with as many entries as there are
         interior and ghost points, in total

  Output parameter:
  array - contiguously allocated array of the appropriate dimension with
          ghost points, pointing to the 1-array

  TODO: 3d version
*/
template <class T>
PetscErrorCode GiveGhostPoints(DM da,T *cgs,void *array)
{
  PetscErrorCode ierr;
  PetscInt       dim;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,&dim,0,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  if (dim == 2) {
    ierr = GiveGhostPoints2d(da,cgs,(T***)array);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Shift indices in an array of type T to endow it with ghost points.
  (e.g. This works for arrays of adoubles or AFields.)

  Input parameters:
  da  - distributed array upon which variables are defined
  cgs - contiguously allocated 1-array with as many entries as there are
        interior and ghost points, in total

  Output parameter:
  a2d - contiguously allocated 2-array with ghost points, pointing to the
        1-array
*/
template <class T>
PetscErrorCode GiveGhostPoints2d(DM da,T *cgs,T **a2d[])
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

/*
  Convert a 2-array defined on a DMDA to a 1-array

  Input parameters:
  da    - distributed array upon which variables are defined
  u     - 2-array to be converted

  Output parameters:
  u_vec - corresponding 1-array

  TODO: Generalise... or just remove
*/
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

/*
  TODO: Documentation
  FIXME and generalise for dimensions and dofs
*/
PetscErrorCode ConvertTo1Array(DM da,void **u,PetscScalar *u_vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ConvertTo1Array2d(da,(PetscScalar**)u,u_vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Create a rectangular sub-identity of the m x m identity matrix.
  than rows n.

  Input parameters:
  n - number of (adjacent) rows to take in slice
  s - starting row index

  Output parameter:
  S - resulting n x m submatrix
*/
template <class T>
PetscErrorCode Subidentity(PetscInt n,PetscInt s,T **S)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    S[i][i+s] = 1.;
  }
  PetscFunctionReturn(0);
}

/*
  Enter unit diagonal to give an identity matrix.

  Input parameter:
  n - number of rows/columns
  I - n x n array with memory pre-allocated
*/
template <class T>
PetscErrorCode Identity(PetscInt n,T **I)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = Subidentity(n,0,I);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
