#include <petscdm.h>
#include <petscdmda.h>
#include <adolc/adolc.h>
#include "contexts.cpp"

extern PetscErrorCode JacobianVectorProduct(Mat A_shell,Vec X,Vec Y);
extern PetscErrorCode JacobianTransposeVectorProduct(Mat A_shell,Vec Y,Vec X);

/*
  ADOL-C implementation for Jacobian vector product, using the forward mode of AD.
  Intended to overload MatMult in matrix-free methods where implicit timestepping
  has been used.

  For an implicit Jacobian we may use the rule that
     G = M*xdot + f(x)    ==>     dG/dx = a*M + df/dx,
  where a = d(xdot)/dx is a constant. Evaluated at x0 and acting upon a vector x1:
     (dG/dx)(x0) * x1 = (a*M + df/dx)(x0) * x1.

  Input parameters:
  A_shell - Jacobian matrix of MatShell type
  X       - vector to be multiplied by A_shell

  Output parameters:
  Y       - product of A_shell and X
*/
PetscErrorCode JacobianVectorProduct(Mat A_shell,Vec X,Vec Y)
{
  MatCtx            *mctx;
  PetscErrorCode    ierr;
  PetscInt          m,n,i;
  const PetscScalar *x0;
  PetscScalar       *action,*x1;
  Vec               localX0,localX1;
  DM                da;

  PetscFunctionBegin;

  /* Get matrix-free context info */
  ierr = MatShellGetContext(A_shell,(void**)&mctx);CHKERRQ(ierr);
  m = mctx->m;
  n = mctx->n;

  /* Get local input vectors and extract data, x0 and x1*/
  ierr = TSGetDM(mctx->ts,&da);CHKERRQ(ierr);

  ierr = DMGetLocalVector(da,&localX0);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,mctx->X,INSERT_VALUES,localX0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,mctx->X,INSERT_VALUES,localX0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX1);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,localX0,&x0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localX1,&x1);CHKERRQ(ierr);

  /* First, calculate action of the -df/dx part using ADOL-C */
  ierr = PetscMalloc1(m,&action);CHKERRQ(ierr);
  fos_forward(tag,m,n,0,x0,x1,NULL,action);     // TODO: Could replace NULL to implement ZOS test

  // TODO: temp --------------------------------------
  PetscInt xs,ys,xm,ym,gxs,gys,gxm,gym,d,j,k = 0;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      for (d=0; d<2; d++) {
        if ((i >= xs) && (i < xs+xm) && (j >= ys) && (j < ys+ym)) {
          ierr = VecSetValuesLocal(Y,1,&k,&action[k],INSERT_VALUES);CHKERRQ(ierr);
        }
        k++;
      }
    }
  }
  // -------------------------------------------------- 
/*
  for (i=0; i<m; i++) {
    ierr = VecSetValuesLocal(Y,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }
*/
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);

  /* Second, shift by action of a*M TODO: Combine this above*/
  ierr = VecAXPY(Y,mctx->shift,X);CHKERRQ(ierr);

  /* Restore local vector */
  ierr = DMDAVecRestoreArray(da,localX1,&x1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localX0,&x0);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX1);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* Intended to overload MatMultTranspose in matrix-free methods */
PetscErrorCode JacobianTransposeVectorProduct(Mat A_shell,Vec Y,Vec X)
{
  MatCtx            *mctx;
  PetscErrorCode    ierr;
  PetscInt          m,n,i;
  const PetscScalar *y0;
  //const PetscScalar *dat_ro;
  PetscScalar       *action,*y1;
  Vec               localY0,localY1;
  DM                da;

  PetscFunctionBegin;

  /* Get matrix-free context info */
  ierr = MatShellGetContext(A_shell,(void**)&mctx);CHKERRQ(ierr);
  m = mctx->m;
  n = mctx->n;

  /* Get local input vectors and extract data, x0 and x1*/
  ierr = TSGetDM(mctx->ts,&da);CHKERRQ(ierr);

  ierr = DMGetLocalVector(da,&localY0);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localY1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,mctx->X,INSERT_VALUES,localY0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,mctx->X,INSERT_VALUES,localY0);CHKERRQ(ierr);
  // TODO: The above should be Y's
  //ierr = DMGlobalToLocalBegin(da,mctx->Y,INSERT_VALUES,localY0);CHKERRQ(ierr);
  //ierr = DMGlobalToLocalEnd(da,mctx->Y,INSERT_VALUES,localY0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,Y,INSERT_VALUES,localY1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,Y,INSERT_VALUES,localY1);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,localY0,&y0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localY1,&y1);CHKERRQ(ierr);

  /* Trace forward using independent variable values    TODO: This should be optional */
  zos_forward(tag,m,n,1,y0,NULL);

  /* Compute action */
  ierr = PetscMalloc1(n,&action);CHKERRQ(ierr);
  fos_reverse(tag,m,n,y1,action);
  ierr = VecRestoreArrayRead(Y,&y0);CHKERRQ(ierr);

  /* Set values in vector */
  for (i=0; i<n; i++) {
    ierr = VecSetValuesLocal(X,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(X);
  ierr = VecAssemblyEnd(X);

  /* Second, shift by action of a*M TODO: Combine this above*/
  ierr = VecAXPY(X,mctx->shift,Y);CHKERRQ(ierr);

  /* Restore local vector */
  ierr = DMDAVecRestoreArray(da,localY1,&y1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localY0,&y0);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localY1);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localY0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
