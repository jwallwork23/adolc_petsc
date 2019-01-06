static char help[] = "Demonstrates adjoint sensitivity analysis for Reaction-Diffusion Equations.\n";

/*
      See ex5.c for details on the equations.
      This code applies the operator overloading automatic differentiation techniques provided by ADOL-C to automatically generate Jacobians for nonlinear partial differential equations (PDEs) and associated adjoint equations. Whilst doing this is unnecessary for equations such as these, where the Jacobian can be derived quite easily, automatic Jacobian generation would be greatly beneficial for more complex PDEs.
      Handcoded Jacobians are included here for comparison.

  Runtime options:
    -forwardonly      - run the forward simulation without adjoint
    -implicitform     - provide IFunction and IJacobian to TS, if not set, RHSFunction and RHSJacobian will be used
    -aijpc            - set the preconditioner matrix to be aij (the Jacobian matrix can be of a different type such as ELL)
    -jacobian_by_hand - Use the hand-coded Jacobian of ex13.c, rather than generating it automatically.
    -no_annotation    - Do not annotate ADOL-C active variables. (Should be used alongside -jacobian_by_hand.)

 */

#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>
#include <adolc/adolc_sparse.h>
#include "utils/jacobian.cxx"

int main(int argc,char **argv)
{
  TS             ts;                  		/* ODE integrator */
  Vec            x,r,xdot;             		/* solution, residual, derivative */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  AdolcCtx       *adctx;
  Vec            lambda[1];
  PetscBool      forwardonly=PETSC_FALSE,implicitform=PETSC_FALSE,byhand=PETSC_FALSE;
  PetscInt       gxs,gys,gxm,gym,i,dofs = 2,ctrl[3] = {0,0,0};
  AField         **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL,**udot_a = NULL,*udot_c = NULL;
  PetscScalar    **Seed = NULL,**Rec = NULL,*u_vec;
  unsigned int   **JP = NULL;
  ISColoring     iscoloring;
  MPI_Comm       comm = MPI_COMM_WORLD;

  ierr = PetscInitialize(&argc,&argv,"petscoptions",help);if (ierr) return ierr;
  ierr = PetscNew(&adctx);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-implicitform",&implicitform,NULL);CHKERRQ(ierr);
  appctx.aijpc = PETSC_FALSE,adctx->no_an = PETSC_FALSE,adctx->sparse = PETSC_FALSE,adctx->sparse_view = PETSC_FALSE;adctx->sparse_view_done = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-aijpc",&appctx.aijpc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse",&adctx->sparse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse_view",&adctx->sparse_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&adctx->no_an,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Jacobian overall",MAT_CLASSID,&appctx.event1);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Sparsity pattern",MAT_CLASSID,&appctx.event2);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Colouring",MAT_CLASSID,&appctx.event3);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Compressed Jac",MAT_CLASSID,&appctx.event4);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("Recovery",MAT_CLASSID,&appctx.event5);CHKERRQ(ierr);
  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;
  appctx.adctx = adctx;

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
  ierr = VecDuplicate(x,&xdot);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  if (!implicitform) {
    ierr = TSSetRHSFunction(ts,NULL,RHSFunctionPassive,&appctx);CHKERRQ(ierr);
  } else {
    ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)IFunctionLocalPassive,&appctx);CHKERRQ(ierr);
  }

  if (!adctx->no_an) {

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Allocate memory for (local) active fields (called AFields) and store 
      references in the application context. The AFields are reused at
      each active section, so need only be created once.

      NOTE: Memory for ADOL-C active variables (such as adouble and AField)
            cannot be allocated using PetscMalloc, as this does not call the
            relevant class constructor. Instead, we use the C++ keyword `new`.

            It is also important to deconstruct and free memory appropriately.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
    adctx->m = dofs*gxm*gym;  // Number of dependent variables
    adctx->n = dofs*gxm*gym;  // Number of independent variables

    // Create contiguous 1-arrays of AFields
    u_c = new AField[gxm*gym];
    f_c = new AField[gxm*gym];
    udot_c = new AField[gxm*gym];

    // Corresponding 2-arrays of AFields
    u_a = new AField*[gym];
    f_a = new AField*[gym];
    udot_a = new AField*[gym];

    // Align indices between array types and endow ghost points
    ierr = GiveGhostPoints(da,u_c,&u_a);CHKERRQ(ierr);
    ierr = GiveGhostPoints(da,f_c,&f_a);CHKERRQ(ierr);
    ierr = GiveGhostPoints(da,udot_c,&udot_a);CHKERRQ(ierr);

    // Store active variables in context
    appctx.u_a = u_a;
    appctx.f_a = f_a;
    appctx.udot_a = udot_a;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Trace function(s) just once
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = PetscMalloc1(adctx->n,&u_vec);CHKERRQ(ierr);
    if (!implicitform) {
      ierr = RHSFunctionActive(ts,1.0,x,r,&appctx);CHKERRQ(ierr);
    } else {
      ierr = IFunction(ts,1.0,x,xdot,r,&appctx);CHKERRQ(ierr);
      ierr = IFunction2(ts,1.0,x,xdot,r,&appctx);CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      In the case where ADOL-C generates the Jacobian in compressed format,
      seed and recovery matrices are required. Since the sparsity structure
      of the Jacobian does not change over the course of the time
      integration, we can save computational effort by only generating
      these objects once.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    if (adctx->sparse) {

      // Generate sparsity pattern
      ierr = PetscMalloc1(adctx->n,&u_vec);CHKERRQ(ierr);
      JP = (unsigned int **) malloc(adctx->m*sizeof(unsigned int*));
      jac_pat(1,adctx->m,adctx->n,u_vec,JP,ctrl);
      if (adctx->sparse_view) {
        ierr = PrintSparsity(comm,adctx->m,JP);CHKERRQ(ierr);
      }

      // Extract coloring
      ierr = GetColoring(da,&iscoloring);CHKERRQ(ierr);
      ierr = CountColors(iscoloring,&adctx->p);CHKERRQ(ierr);

      // Generate seed matrix
      ierr = AdolcMalloc2(adctx->n,adctx->p,&Seed);CHKERRQ(ierr);
      ierr = GenerateSeedMatrix(iscoloring,Seed);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
      if (adctx->sparse_view) {
        ierr = PrintMat(comm,"Seed matrix:",adctx->n,adctx->p,Seed);CHKERRQ(ierr);
      }

      // Generate recovery matrix
      ierr = AdolcMalloc2(adctx->m,adctx->p,&Rec);CHKERRQ(ierr);
      ierr = GetRecoveryMatrix(Seed,JP,adctx->m,adctx->p,Rec);CHKERRQ(ierr);

      // Store results and free workspace
      adctx->Rec = Rec;
      for (i=0;i<adctx->m;i++)
        free(JP[i]);
      free(JP);
      ierr = PetscFree(u_vec);CHKERRQ(ierr);
    } else {
      adctx->p = adctx->n;
      ierr = AdolcMalloc2(adctx->n,adctx->p,&Seed);CHKERRQ(ierr);
      ierr = Identity(adctx->n,Seed);CHKERRQ(ierr);
    }
    adctx->Seed = Seed;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!implicitform) {
    if (!byhand) {
      ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobianAdolc,&appctx);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobianByHand,&appctx);CHKERRQ(ierr);
    }
  } else {
    if (appctx.aijpc) {
      Mat                    A,B;

      ierr = DMSetMatType(da,MATSELL);CHKERRQ(ierr);
      ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
      ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      /* FIXME do we need to change viewer to display matrix in natural ordering as DMCreateMatrix_DA does? */
      if (!byhand) {
        ierr = TSSetIJacobian(ts,A,B,IJacobianAdolc,&appctx);CHKERRQ(ierr);
      } else {
        ierr = TSSetIJacobian(ts,A,B,IJacobianByHand,&appctx);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&A);CHKERRQ(ierr);
      ierr = MatDestroy(&B);CHKERRQ(ierr);
    } else {
      if (!byhand) {
        ierr = TSSetIJacobian(ts,NULL,NULL,IJacobianAdolc,&appctx);CHKERRQ(ierr);
      } else {
        ierr = TSSetIJacobian(ts,NULL,NULL,IJacobianByHand,&appctx);CHKERRQ(ierr);
      }
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(da,x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Have the TS save its trajectory so that TSAdjointSolve() may be used
    and set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!forwardonly) {
    ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,200.0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,0.5);CHKERRQ(ierr);
  } else {
    ierr = TSSetMaxTime(ts,2000.0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,10);CHKERRQ(ierr);
  }
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  if (!forwardonly) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Start the Adjoint model
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = VecDuplicate(x,&lambda[0]);CHKERRQ(ierr);
    /*   Reset initial conditions for the adjoint integration */
    ierr = InitializeLambda(da,lambda[0],0.5,0.5);CHKERRQ(ierr);
    ierr = TSSetCostGradients(ts,1,lambda,NULL);CHKERRQ(ierr);
    ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
    ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space and call destructors for AFields.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if (!adctx->no_an) {
    if (adctx->sparse)
      ierr = AdolcFree2(Rec);CHKERRQ(ierr);
    ierr = AdolcFree2(Seed);CHKERRQ(ierr);
    udot_a += gys;
    f_a += gys;
    u_a += gys;
    delete[] udot_c;
    delete[] udot_a;
    delete[] f_a;
    delete[] u_a;
    delete[] f_c;
    delete[] u_c;
  }
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFree(adctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex !single

   TODO

TEST*/
