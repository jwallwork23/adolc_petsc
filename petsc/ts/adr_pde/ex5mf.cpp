static char help[] = "Demonstrates automatic, matrix-free Jacobian generation using ADOL-C for a time-dependent PDE in 2d, solved using implicit timestepping.\n";

/*
  See ex5.c for details on the equation.

  Here implicit Crank-Nicolson timestepping is used to solve the same problem as in ex5.c. Another
  key difference is that functions are calculated in a local sense, using the local implementations
  IFunctionLocalPassive and IFunctionLocalActive, as in ex5imp. These are passed to the TS solver
  using DMTSSetIFunctionLocal. The Jacobian is generated matrix-free using MyMult, which overloads the
  MatMult operation. The function IJacobian acts to pass TS context information to the matrix-free
  context.
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>            // Includes ADOL-C
#include "../../utils/matfree.cpp"  // Includes context structures and matrix free drivers
#include "utils/jacobian.cpp"

#define tag 1

int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x,r;                 /* solution, residual */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;              /* Application context */
  MatCtx         matctx;              /* Matrix (free) context */
  AdolcCtx       *adctx;
  Mat            A;                   /* (Matrix free) Jacobian matrix */
  PetscInt       gys,gxm,gym;
  AField         **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char*)0,help);
  PetscFunctionBeginUser;
  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,65,65,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"v");CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create matrix free context
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetMatType(da,MATSHELL);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatShellSetContext(A,&matctx);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A,MATOP_MULT,(void (*)(void))JacobianVectorProduct);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&matctx.X);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&matctx.Xdot);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)IFunctionLocalPassive,&appctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Allocate memory for (local) active fields (called AFields) and store 
    references in the application context. The AFields are reused at
    each active section, so need only be created once.

    NOTE: Memory for ADOL-C active variables (such as adouble and AField)
          cannot be allocated using PetscMalloc, as this does not call the
          relevant class constructor. Instead, we use the C++ keyword `new`.

          It is also important to deconstruct and free memory appropriately.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDAGetGhostCorners(da,NULL,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  matctx.m = 2*gxm*gym;
  matctx.n = 2*gxm*gym;

  // Create contiguous 1-arrays of AFields
  u_c = new AField[gxm*gym];
  f_c = new AField[gxm*gym];

  // Corresponding 2-arrays of AFields
  u_a = new AField*[gym];
  f_a = new AField*[gym];

  // Align indices between array types and endow ghost points
  ierr = AFieldGiveGhostPoints2d(da,u_c,&u_a);CHKERRQ(ierr);
  ierr = AFieldGiveGhostPoints2d(da,f_c,&f_a);CHKERRQ(ierr);

  // Store active variables in context
  appctx.u_a = u_a;
  appctx.f_a = f_a;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian. In this case, IJacobian simply acts to pass context
     information to the matrix-free Jacobian vector product.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetIJacobian(ts,A,A,IJacobianMatFree,&appctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(da,x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace function just once
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscMalloc1(1,&adctx);CHKERRQ(ierr);
  adctx->no_an = PETSC_FALSE;appctx.adctx = adctx;
  ierr = IFunction(ts,0.,x,matctx.Xdot,r,&appctx);CHKERRQ(ierr);
  ierr = PetscFree(adctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,2000.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,10);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&matctx.X);CHKERRQ(ierr);
  ierr = VecDestroy(&matctx.Xdot);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  f_a += gys;
  u_a += gys;
  delete[] f_a;
  delete[] u_a;
  delete[] f_c;
  delete[] u_c;
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
