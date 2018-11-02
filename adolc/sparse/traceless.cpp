#define ADOLC_TAPELESS
#include <adolc/adtl.h>
using namespace adtl;
#include "example_utils.cpp"

extern PetscErrorCode JacobianVectorProduct(Mat J,Vec x,Vec y);

int main(int argc,char **args)
{
  AdolcCtx        ctx;
  PetscErrorCode  ierr;
  MPI_Comm        comm = MPI_COMM_WORLD;
  PetscInt        n = 6,m = 3,i,mi[m],ni[n];
  PetscScalar     x[n],y[m];
  Vec             W,X,Y;
  Mat             J;

  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;

  /* Give values for independent variables */
  for(i=0;i<n;i++) {
    x[i] = log(1.0+i);
    ni[i] = i;
  }

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

  /* Create matrix free matrix */
  ierr = MatCreateShell(comm,m,n,m,n,NULL,&J);CHKERRQ(ierr);
  ierr = MatShellSetContext(J,&ctx);CHKERRQ(ierr);
  ierr = MatShellSetOperation(J,MATOP_MULT,(void(*)(void))JacobianVectorProduct);CHKERRQ(ierr);

  /*
    Evaluate Jacobian vector product matrix free:
                  W = J * X
  */
  ierr = VecCreate(comm,&W);CHKERRQ(ierr);
  ierr = VecSetSizes(W,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(W);CHKERRQ(ierr);
  ierr = MatMult(J,X,W);CHKERRQ(ierr);

  /* Print results */
  ierr = PetscPrintf(comm,"\nFunction evaluation : \n");CHKERRQ(ierr);
  ierr = VecView(Y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"\nJacobian vector product :\n");CHKERRQ(ierr);
  ierr = VecView(W,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

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
  PetscScalar       *action,*dat,*ydat;
  adouble           xad[n],yad[m];

  PetscFunctionBegin;

  /* Read data and allocate memory */
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&dat);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&action);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&dat_ro);CHKERRQ(ierr);
  ierr = VecGetArray(y,&ydat);CHKERRQ(ierr);

  /* Call function c(x) and apply Jacobian vector product*/
  for(i=0; i<n; i++) {
    xad[i] = dat_ro[i];
    xad[i].setADValue(dat_ro[i]);
  }
  ierr = ActiveEvaluate(xad,yad);CHKERRQ(ierr);
  for(i=0; i<m; i++) {
    ydat[i] = yad[i].getValue();
    action[i] = yad[i].getADValue();
  }
  for (i=0; i<m; i++) {
    ierr = VecSetValues(y,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Restore arrays and free memory */
  ierr = VecRestoreArray(y,&ydat);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x,&dat_ro);CHKERRQ(ierr);
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = PetscFree(dat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

