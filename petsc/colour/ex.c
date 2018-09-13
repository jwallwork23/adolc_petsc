#include <petscsnes.h>

/* Some arbitrary application context format */
typedef struct {
  PetscReal c;
} AppCtx;

PetscErrorCode FormJacobian(SNES snes,Vec U,Mat J,Mat Jpre,void *ctx){

  PetscErrorCode ierr;

  PetscFunctionBegin;

  // TODO

  PetscFunctionReturn(0);
}


int main(int argc,char **args)
{
  PetscErrorCode ierr;
  AppCtx         appctx;
  SNES           snes;
  Mat            J;
  Vec            u;
  ISColoring     iscoloring;
  MatFDColoring  fdcoloring;
  MatColoring    coloring;

  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;

  /* Initialise the nonzero structure of the Jacobian, using ADOL-C */
  ierr = FormJacobian(snes,u,&J,&J,&appctx);CHKERRQ(ierr);

  /*
    Colour the matrix, i.e. determine groups of columns that share no common rows. These columns
    in the Jacobian can all be computed simultaneously.
  */
  ierr = MatColoringCreate(J,&coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);	// Use 'smallest last' type
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring,&iscoloring);CHKERRQ(ierr);

  ierr = MatFDColoringCreate(J,iscoloring,&fdcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode(*)(void))FormFunction,&appctx);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);

  /* Compute Jacobians this way using SNES */
  ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,fdcoloring);CHKERRQ(ierr);

  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);

  ierr = PetscFinalize();

  return ierr;
}
