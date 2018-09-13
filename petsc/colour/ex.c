#include <petscsnes.h>

/* Some arbitrary application context format */
typedef struct {
  PetscReal c;
} AppCtx;

/*
  FormJacobian1 from snes ex1

  TODO: Consider something that actually has sparsity
*/
PetscErrorCode FormJacobian(SNES snes,Vec U,Mat J,Mat Jpre,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u;
  PetscScalar       A[2][2];
  PetscInt          idx[2] = {0,1};

  PetscFunctionBegin;

  /* Get pointer to vector data */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);

  /* Compute Jacobian entries and insert into matrix */
  A[0][0] = 2.0*u[0]+u[1];
  A[0][1] = u[0];
  A[1][0] = u[1];
  A[1][1] = u[0]+2.0*u[1];
  ierr = MatSetValues(Jpre,2,idx,2,idx,A,INSERT_VALUES);CHKERRQ(ierr);

  /* Restore vector */
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);

  /* Assemble matrix */
  ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

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
  ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);	// Use 'smallest last' method
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
