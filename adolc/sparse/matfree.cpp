#include <adolc/adolc.h>
#include "example_utils.cpp"

#define tag 1

extern PetscErrorCode JacobianVectorProduct(Mat J,Vec x,Vec y);
extern PetscErrorCode JacobianTransposeVectorProduct(Mat J,Vec y,Vec x);

typedef struct {
  PetscScalar *indep_vals; // Need to provide a 1-array containing independent variable values
  PetscBool   trace;       // Toggle whether or not to trace forward, thereby writing to tape
} AdolcCtx;

int main(int argc,char **args)
{
  AdolcCtx          ctx;
  PetscErrorCode  ierr;
  MPI_Comm        comm = MPI_COMM_WORLD;
  PetscInt        n = 6,m = 3,i,j,mi[m],ni[n];
  PetscScalar     x[n],y[m];
  adouble         xad[n],yad[m];
  Vec             W,X,Y,Z;
  Mat             J;

  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;

  /* Give values for independent variables and put these in context */
  for(i=0;i<n;i++) {
    x[i] = log(1.0+i);
    ni[i] = i;
  }
  ctx.indep_vals = x;
  ctx.trace = PETSC_TRUE;

  /* Trace function c(x) */
  trace_on(tag);
    for(i=0;i<n;i++)
      xad[i] <<= x[i];

    ierr = ActiveEvaluate(xad,yad);CHKERRQ(ierr);

    for(i=0;i<m;i++)
      yad[i] >>= y[i];
  trace_off();

  /* Function evaluation as above */
  ierr = PetscPrintf(comm,"\n Function evaluation by RHS : ");CHKERRQ(ierr);
  for(j=0;j<m;j++) {
    ierr = PetscPrintf(comm," %e ",y[j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);

  /* Trace over ZOS to check function evaluation and enable reverse mode */
  zos_forward(tag,m,n,1,x,y);
  ierr = PetscPrintf(comm,"\n Function evaluation by ZOS : ");CHKERRQ(ierr);
  for(j=0;j<m;j++) {
    ierr = PetscPrintf(comm," %e ",y[j]);CHKERRQ(ierr);
    mi[j] = j;
  }
  ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);

  /* Insert independent variable values into a Vec */
  ierr = VecCreate(comm,&X);CHKERRQ(ierr);
  ierr = VecSetSizes(X,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(X);CHKERRQ(ierr);
  ierr = VecSetValues(X,n,ni,x,INSERT_VALUES);CHKERRQ(ierr);
  //ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Insert dependent variable values into a Vec */
  ierr = VecCreate(comm,&Y);CHKERRQ(ierr);
  ierr = VecSetSizes(Y,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(Y);CHKERRQ(ierr);
  ierr = VecSetValues(Y,m,mi,y,INSERT_VALUES);CHKERRQ(ierr);
  //ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Create matrix free matrix */
  ierr = MatCreateShell(comm,m,n,m,n,NULL,&J);CHKERRQ(ierr);
  ierr = MatShellSetContext(J,&ctx);CHKERRQ(ierr);
  ierr = MatShellSetOperation(J,MATOP_MULT,(void(*)(void))JacobianVectorProduct);CHKERRQ(ierr);
  ierr = MatShellSetOperation(J,MATOP_MULT_TRANSPOSE,(void(*)(void))JacobianTransposeVectorProduct);CHKERRQ(ierr);

  /*
    Evaluate Jacobian vector product matrix free:
                  W = J * X
  */
  ierr = VecCreate(comm,&W);CHKERRQ(ierr);
  ierr = VecSetSizes(W,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(W);CHKERRQ(ierr);
  ierr = MatMult(J,X,W);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"\nJacobian vector product:\n");CHKERRQ(ierr);
  ierr = VecView(W,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
    Evaluate Jacobian transpose vector product matrix free:
                 Z = J^T * Y
  */
  ierr = VecCreate(comm,&Z);CHKERRQ(ierr);
  ierr = VecSetSizes(Z,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(Z);CHKERRQ(ierr);
  ierr = MatMultTranspose(J,Y,Z);CHKERRQ(ierr);	// Note: This has been overloaded for matrix J
  ierr = PetscPrintf(comm,"\nJacobian transpose vector product:\n");CHKERRQ(ierr);
  ierr = VecView(Z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Clear workspace and finalise */
  ierr = VecDestroy(&Z);CHKERRQ(ierr);
  ierr = VecDestroy(&W);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return ierr;
}

/*
  ADOL-C implementation for Jacobian vector product, using the forward mode of AD.
  Intended to overload MatMult in matrix-free methods

  Input parameters:
  J - Jacobian matrix of MatShell type
  x - vector to be multiplied by J

  Output parameters:
  y - product of J and x
*/
PetscErrorCode JacobianVectorProduct(Mat J,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PetscInt          m,n,i;
  const PetscScalar *dat_ro;
  PetscScalar       *action,*dat;

  PetscFunctionBegin;

  /* Read data and allocate memory */
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&dat);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&action);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&dat_ro);CHKERRQ(ierr);
  for (i=0; i<n; i++)
    dat[i] = dat_ro[i]; // FIXME: How to avoid this conversion from read only?

  /* Compute action of Jacobian on vector */
  fos_forward(tag,m,n,0,dat,dat,NULL,action);
  ierr = VecRestoreArrayRead(x,&dat_ro);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = VecSetValues(y,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = PetscFree(dat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  ADOL-C implementation for Jacobian transpose vector product, using reverse mode of AD.
  Intended to overload MatMultTranspose in matrix-free methods.

  Input parameters:
  J - MatShell type Jacobian matrix
  y - vector to be multiplied
  x - Jacobian transpose vector product
*/
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
  ierr = PetscMalloc1(m,&dat);CHKERRQ(ierr);
  ierr = VecGetArrayRead(y,&dat_ro);CHKERRQ(ierr);

  /* Trace forward using independent variable values */
  ierr = MatShellGetContext(J,&ctx);CHKERRQ(ierr);
  if (ctx->trace)
    zos_forward(tag,m,n,1,ctx->indep_vals,NULL);

  /* Compute action */
  ierr = PetscMalloc1(n,&action);CHKERRQ(ierr);
  for (i=0; i<m; i++)
    dat[i] = dat_ro[i]; // TODO: How to avoid this conversion from read only?
  fos_reverse(tag,m,n,dat,action);
  ierr = VecRestoreArrayRead(y,&dat_ro);CHKERRQ(ierr);

  /* Set values in vector */
  for (i=0; i<n; i++) {
    ierr = VecSetValues(x,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = PetscFree(dat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

