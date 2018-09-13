#include <petscsnes.h>

/* Some arbitrary application context format */
typedef struct {
  PetscReal c;
} AppCtx;

/*
  FormFunction1 from snes ex1

  TODO: Consider something that actually has sparsity
*/
PetscErrorCode FormFunction(SNES snes,Vec U,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u;
  PetscScalar       *f;

  /*
   Get pointers to vector data.
      - For default PETSc vectors, VecGetArray() returns a pointer to
        the data array.  Otherwise, the routine is implementation dependent.
      - You MUST call VecRestoreArray() when you no longer need access to
        the array.
   */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /* Compute function */
  f[0] = u[0]*u[0] + u[0]*u[1] - 3.0;
  f[1] = u[0]*u[1] + u[1]*u[1] - 6.0;

  /* Restore vectors */
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  return 0;
}

/*
  FormJacobian1 from snes ex1

  TODO: Consider something that actually has sparsity
*/
PetscErrorCode FormJacobian(SNES snes,Vec U,Mat J,Mat Jpre,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u;
  PetscScalar       A[4];
  PetscInt          idx[2] = {0,1};

  PetscFunctionBegin;

  /* Get pointer to vector data */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);

  /* Compute Jacobian entries and insert into matrix */
  A[0] = 2.0*u[0]+u[1];
  A[1] = u[0];
  A[2] = u[1];
  A[3] = u[0]+2.0*u[1];
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
  Vec            u,r;
  ISColoring     iscoloring;
  MatFDColoring  fdcoloring;
  MatColoring    coloring;

  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;

  /* Create nonlinear solver context */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* Create vectors for solution and nonlinear function */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&r);CHKERRQ(ierr);

  /* Create Jacobian object */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);

  /* Initialise the nonzero structure of the Jacobian TODO: use ADOL-C */
  ierr = FormJacobian(snes,u,J,J,&appctx);CHKERRQ(ierr);

  /*
    Colour the matrix, i.e. determine groups of columns that share no common rows. These columns
    in the Jacobian can all be computed simultaneously.
  */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Creating colouring of J...\n");
  ierr = MatColoringCreate(J,&coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);	// Use 'smallest last' method
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring,&iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode(*)(void))FormFunction,&appctx);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  //ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);

  // MORE TODO: See mat/ex16

  /* Compute Jacobians this way using SNES */
  ierr = SNESSetJacobian(snes,J,J,SNESComputeJacobianDefaultColor,fdcoloring);CHKERRQ(ierr);

  /* Free workspace */
  ierr = MatFDColoringDestroy(&fdcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return ierr;
}
