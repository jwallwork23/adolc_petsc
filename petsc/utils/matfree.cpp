#include <petscts.h>
#include <adolc/adolc.h>

#define tag 1	// TODO: Generalise to case where multiple tags may be used

extern PetscErrorCode JacobianVectorProduct(Mat J,Vec x,Vec y);
extern PetscErrorCode JacobianTransposeVectorProduct(Mat J,Vec y,Vec x);

typedef struct {
  PetscScalar *indep_vals; // Need to provide a 1-array containing independent variable values
  PetscBool   trace;       // Toggle whether or not to trace forward, thereby writing to tape
} AdolcCtx;

/*@C
  ADOL-C implementation for Jacobian vector product, using the forward mode of AD.
  Intended to overload MatMult in matrix-free methods

  Input parameters:
  J - Jacobian matrix of MatShell type
  x - vector to be multiplied by J

  Output parameters:
  y - product of J and x

  TODO: Update from ex5mf and use this version
@*/
PetscErrorCode JacobianVectorProduct(Mat J,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PetscInt          m,n,i;
  const PetscScalar *dat_ro;
  PetscScalar       *action,*dat;

  PetscFunctionBegin;

  /* Read data and allocate memory */
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&dat_ro);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&dat);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&action);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&dat_ro);CHKERRQ(ierr);
  for (i=0; i<n; i++)
    dat[i] = dat_ro[i];	// FIXME: How to avoid this conversion from read only?

  /* Compute action of Jacobian on vector */
  fos_forward(tag,m,n,0,dat,dat,NULL,action);
  ierr = VecRestoreArrayRead(x,&dat_ro);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = VecSetValues(y,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = PetscFree(dat);CHKERRQ(ierr);
  ierr = PetscFree(dat_ro);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  ADOL-C implementation for Jacobian transpose vector product, using reverse mode of AD.
  Intended to overload MatMultTranspose in matrix-free methods.

  Input parameters:
  J - MatShell type Jacobian matrix
  y - vector to be multiplied
  x - Jacobian transpose vector product
@*/
/* Intended to overload MatMultTranspose in matrix-free methods */
PetscErrorCode JacobianTransposeVectorProduct(Mat J,Vec y,Vec x)
{
  AdolcCtx            *ctx;
  PetscErrorCode    ierr;
  PetscInt          i,m,n;
  const PetscScalar *dat_ro;
  PetscScalar       *action,*dat;

  PetscFunctionBegin;

  /* Read data and allocate memory */
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&dat_ro);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&dat);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y,&dat_ro);CHKERRQ(ierr);

  /* Trace forward using independent variable values */
  ierr = MatShellGetContext(J,&ctx);CHKERRQ(ierr);
  if (ctx->trace)
    zos_forward(tag,m,n,1,ctx->indep_vals,NULL);

  /* Compute action */
  ierr = PetscMalloc1(n,&action);CHKERRQ(ierr);
  for (i=0; i<m; i++)
    dat[i] = dat_ro[i];	// TODO: How to avoid this conversion from read only?
  fos_reverse(tag,m,n,dat,action);
  ierr = VecRestoreArrayRead(y,&dat_ro);CHKERRQ(ierr);

  /* Set values in vector */
  for (i=0; i<n; i++) {
    ierr = VecSetValues(x,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = PetscFree(dat);CHKERRQ(ierr);
  ierr = PetscFree(dat_ro);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
