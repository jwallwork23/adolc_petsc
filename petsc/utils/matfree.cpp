#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>

#define tag 1	// TODO: Generalise to case where multiple tags may be used

/*@C
  ADOL-C implementation for Jacobian vector product, using the forward mode of AD.
  Intended to overload MatMult in matrix-free methods

  Input parameters:
  J - Jacobian matrix of MatShell type
  x - vector to be multiplied by J

  Output parameters:
  y - product of J and x
@*/
PetscErrorCode JacobianVectorProduct(Mat J,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PetscInt          m,n;
  const PetscScalar *dat_ro;
  PetscScalar       *action,*dat,*val;

  PetscFunctionBegin;

  /* Read data and allocate memory */
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&dat_ro);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&dat);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&val);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&action);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&dat_ro);CHKERRQ(ierr);
  for (i=0; i<n; i++)
    dat[i] = dat_ro[i];	// FIXME: How to avoid this conversion from read only?

  /* Compute action of Jacobian on vector */
  fos_forward(tag,m,n,0,dat,dat,val,action);
  ierr = VecRestoreArrayRead(x,&dat_ro);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = VecSetValues(y,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = PetscFree(val);CHKERRQ(ierr);
  ierr = PetscFree(dat);CHKERRQ(ierr);
  ierr = PetscFree(dat_ro);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  ADOL-C implementation for Jacobian transpose vector product, using reverse mode of AD.
  Intended to overload MatMultTranspose in matrix-free methods.

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
    dat[i] = dat_ro[i];	// TODO: How to avoid this conversion from read only?
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
