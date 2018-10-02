#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>
#include "allocation.cpp"

#define tag 1	// TODO: Generalise to case where multiple tags may be used

/*@C
  Jacobian transpose vector product, provided matrix free using reverse mode of AD

  Input parameters:
  J - MatShell type Jacobian matrix
  x - vector to be multiplied
  y - Jacobian transpose vector product
@*/
/* Intended to overload MatMultTranspose in matrix-free methods */
PetscErrorCode JacobianTransposeVectorProduct(Mat J,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PetscInt          i,m,n;
  const PetscScalar *dat_ro;
  PetscScalar       *action,*dat;

  PetscFunctionBegin;

  // TODO: How to call zos_forward here?

  /* Read data and allocate memory */
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&dat_ro);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&dat);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&dat_ro);CHKERRQ(ierr);

  /* Compute action */
  ierr = PetscMalloc1(n,&action);CHKERRQ(ierr);
  for (i=0; i<m; i++)
    dat[i] = dat_ro[i];
  fos_reverse(tag,m,n,uu,action);
  ierr = VecRestoreArrayRead(x,&dat_ro);CHKERRQ(ierr);

  /* Set values in vector */
  for (i=0; i<n; i++) {
    ierr = VecSetValues(y,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = PetscFree(dat);CHKERRQ(ierr);
  ierr = PetscFree(dat_ro);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
