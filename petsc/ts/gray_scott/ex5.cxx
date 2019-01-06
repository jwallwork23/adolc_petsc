static char help[] = "Demonstrates automatic Jacobian generation using ADOL-C for a time-dependent PDE in 2d, solved using explicit timestepping.\n";

/*
      See ex5.c for details on the equations.
      This code applies the operator overloading automatic differentiation techniques provided by ADOL-C to automatically generate Jacobians for nonlinear partial differential equations (PDEs). Whilst this is unnecessary for equations such as these, where the Jacobian can be derived quite easily, automatic Jacobian generation would be greatly beneficial for more complex PDEs.
      Handcoded Jacobians are included here for comparison.
*/

/*
      Helpful runtime monitor options:
           -ts_monitor_draw_solution
           -draw_save <filename>

      Helpful runtime monitor options for debugging:
           -da_grid_x 12 -da_grid_y 12 -ts_max_steps 1 -snes_test_jacobian
           -da_grid_x 12 -da_grid_y 12 -ts_max_steps 1 -snes_test_jacobian -snes_test_jacobian_view

      Command line arguments to test different aspects of the automatic
      differentiation process.

      -adolc_test_zos      : Verify zero order evalutation in ADOL-C gives
                             the same result as evalutating the RHS.
      -adolc_test_zos_view : View the zero order evaluation component-by-
                             component.
      -adolc_sparse        : Assemble the Jacobian using ADOL-C sparse
                             drivers and PETSc colouring routines.
      -adolc_sparse_view   : Print the matrices involved in the sparse
                             Jacobian computation.
      -jacobian_by_hand    : Use the hand-coded Jacobian of ex13.c, rather
                             than generating it automatically.
      -no_annotation       : Do not annotate ADOL-C active variables.
                             (Should be used alongside -jacobian_by_hand.)
*/

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscts.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>	// Include ADOL-C
#include <adolc/adolc_sparse.h> // Include ADOL-C sparse drivers
#include "utils/jacobian.cxx"

int main(int argc,char **argv)
{
  TS             ts;                    /* ODE integrator */
  Vec            x,r;                   /* solution, residual */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  AdolcCtx       *adctx;
  PetscInt       gxs,gys,gxm,gym,i,dofs = 2,ctrl[3] = {0,0,0};
  AField         **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;
  PetscScalar    **Seed = NULL,**Rec = NULL,*u_vec;
  unsigned int   **JP = NULL;
  ISColoring     iscoloring;
  PetscBool      byhand = PETSC_FALSE;
  MPI_Comm       comm = MPI_COMM_WORLD;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,"petscoptions",help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  ierr = PetscNew(&adctx);CHKERRQ(ierr);
  adctx->zos = PETSC_FALSE;adctx->zos_view = PETSC_FALSE;adctx->no_an = PETSC_FALSE;adctx->sparse = PETSC_FALSE;adctx->sparse_view = PETSC_FALSE;adctx->sparse_view_done = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos",&adctx->zos,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos_view",&adctx->zos_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse",&adctx->sparse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse_view",&adctx->sparse_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&adctx->no_an,NULL);CHKERRQ(ierr);
  appctx.D1     = 8.0e-5;
  appctx.D2     = 4.0e-5;
  appctx.gamma  = .024;
  appctx.kappa  = .06;
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context and set problem RHS
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunctionPassive,&appctx);CHKERRQ(ierr);

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

    // Corresponding 2-arrays of AFields
    u_a = new AField*[gym];
    f_a = new AField*[gym];

    // Align indices between array types and endow ghost points
    ierr = GiveGhostPoints(da,u_c,&u_a);CHKERRQ(ierr);
    ierr = GiveGhostPoints(da,f_c,&f_a);CHKERRQ(ierr);

    // Store active variables in context
    appctx.u_a = u_a;
    appctx.f_a = f_a;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Trace function just once
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = RHSFunctionActive(ts,1.0,x,r,&appctx);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      In the case where ADOL-C generates the Jacobian in compressed format,
      seed and recovery matrices are required. Since the sparsity structure
      of the Jacobian does not change over the course of the time
      integration, we can save computational effort by only generating
      these objects once.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    if (adctx->sparse) {

      // Extract colouring
      ierr = GetColoring(da,&iscoloring);CHKERRQ(ierr);
      ierr = CountColors(iscoloring,&adctx->p);CHKERRQ(ierr);

      // Generate sparsity pattern and create an associated colouring
      ierr = PetscMalloc1(adctx->n,&u_vec);CHKERRQ(ierr);
      JP = (unsigned int **) malloc(adctx->m*sizeof(unsigned int*));
      printf("m = %d, n = %d\n",adctx->m,adctx->n);
      jac_pat(1,adctx->m,adctx->n,u_vec,JP,ctrl);

      ierr = DMGetSparsity(da,JP);CHKERRQ(ierr);

      // TODO: Avoid need for adolc_sparse using the below
      //ierr = GenerateSparsityPattern(iscoloring,JP);CHKERRQ(ierr);

      if (adctx->sparse_view) {
        ierr = PrintSparsity(comm,adctx->m,JP);CHKERRQ(ierr);
      }

      // Generate seed matrix
      ierr = AdolcMalloc2(adctx->n,adctx->p,&Seed);CHKERRQ(ierr);
      ierr = GenerateSeedMatrix(iscoloring,Seed);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
      if (adctx->sparse_view) {
        ierr = PrintMat(comm,"Seed matrix:",adctx->n,adctx->p,Seed);CHKERRQ(ierr);
      }

      // Generate recovery matrix
      ierr = AdolcMalloc2(adctx->m,adctx->p,&Rec);
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

    /*
      Printing for ZOS test
    */
    if (adctx->zos)
      ierr = PetscPrintf(comm,"    If ||F_zos(x) - F_rhs(x)||_2/||F_rhs(x)||_2 is O(1.e-8), ADOL-C function evaluation\n      is probably correct.\n");CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!byhand) {
    ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobianAdolc,&appctx);CHKERRQ(ierr);
  } else {
    ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobianByHand,&appctx);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(da,x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,2000.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.0001);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space and call destructors for AFields.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if (!adctx->no_an) {
    if (adctx->sparse)
      ierr = AdolcFree2(Rec);CHKERRQ(ierr);
    ierr = AdolcFree2(Seed);CHKERRQ(ierr);
    f_a += gys;
    u_a += gys;
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

   test:
      args: -ts_view  -ts_monitor -ts_max_time 500
      requires: double
      timeoutfactor: 3

   test:
      suffix: 2
      args: -ts_view  -ts_monitor -ts_max_time 500 -ts_monitor_draw_solution
      requires: x double
      output_file: output/ex5_1.out
      timeoutfactor: 3

TEST*/
