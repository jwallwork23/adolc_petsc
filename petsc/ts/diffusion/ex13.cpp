static char help[] = "Demonstrates automatic Jacobian computation using ADOL-C for a time-dependent PDE in 2d. \n";
/*
   u_t = uxx + uyy
   0 < x < 1, 0 < y < 1;
   At t=0: u(x,y) = exp(c*r*r*r), if r=PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5)) < .125
           u(x,y) = 0.0           if r >= .125

    mpiexec -n 2 ./ex13 -da_grid_x 40 -da_grid_y 40 -ts_max_steps 2 -snes_monitor -ksp_monitor
    mpiexec -n 1 ./ex13 -snes_fd_color -ts_monitor_draw_solution
    mpiexec -n 2 ./ex13 -ts_type sundials -ts_monitor 

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

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>	// Include ADOL-C
#include <adolc/adolc_sparse.h>
#include "utils/jacobian.cpp"


int main(int argc,char **argv)
{
  TS             ts;                    /* nonlinear solver */
  Vec            u,r;                   /* solution, residual vector */
  Mat            J;                     /* Jacobian matrix */
  PetscInt       steps,gxs,gys,gxm,gym,i,ctrl[3] = {0,0,0};
  PetscErrorCode ierr;
  DM             da;
  PetscReal      ftime,dt;
  AppCtx         user;                  /* user-defined work context */
  AdolcCtx       *adctx;
  adouble        **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;  /* active variables */
  PetscScalar    **Seed = NULL,**Rec = NULL,*u_vec;
  unsigned int   **JP = NULL;
  ISColoring     iscoloring;
  PetscBool      byhand = PETSC_FALSE;
  MPI_Comm       comm = MPI_COMM_WORLD;

  ierr = PetscInitialize(&argc,&argv,"petscoptions",help);if (ierr) return ierr;
  ierr = PetscMalloc1(1,&adctx);CHKERRQ(ierr);
  adctx->no_an = PETSC_FALSE;adctx->zos = PETSC_FALSE;adctx->zos_view = PETSC_FALSE;adctx->sparse = PETSC_FALSE;adctx->sparse_view = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos",&adctx->zos,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos_view",&adctx->zos_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse",&adctx->sparse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse_view",&adctx->sparse_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&adctx->no_an,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(comm,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,8,8,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&r);CHKERRQ(ierr);

  /* Initialize user application context */
  user.c = -30.0;
  user.adctx = adctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,r,RHSFunctionPassive,&user);CHKERRQ(ierr);

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
    adctx->m = gxm*gym;  // Number of dependent variables
    adctx->n = gxm*gym;  // Number of independent variables

    // Create contiguous 1-arrays of AFields
    u_c = new adouble[gxm*gym];
    f_c = new adouble[gxm*gym];

    // Corresponding 2-arrays of AFields
    u_a = new adouble*[gym];
    f_a = new adouble*[gym];

    // Align indices between array types and endow ghost points
    ierr = GiveGhostPoints(da,u_c,&u_a);CHKERRQ(ierr);
    ierr = GiveGhostPoints(da,f_c,&f_a);CHKERRQ(ierr);

    // Store active variables in context
    user.u_a = u_a;
    user.f_a = f_a;

    // Trace RHSFunction, so that ADOL-C has tape to read from
    ierr = RHSFunctionActive(ts,1.0,u,r,&user);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      In the case where ADOL-C generates the Jacobian in compressed format,
      seed and recovery matrices are required. Since the sparsity structure
      of the Jacobian does not change over the course of the time
      integration, we can save computational effort by only generating
      these objects once.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    if (adctx->sparse) {

      // Generate sparsity pattern and create an associated colouring
      ierr = PetscMalloc1(adctx->n,&u_vec);CHKERRQ(ierr);
      JP = (unsigned int **) malloc(adctx->m*sizeof(unsigned int*));
      jac_pat(1,adctx->m,adctx->n,u_vec,JP,ctrl);
      if (user.adctx->sparse_view) {
        ierr = PrintSparsity(comm,adctx->m,JP);CHKERRQ(ierr);
      }

      // Extract colouring
      ierr = GetColoring(da,&iscoloring);CHKERRQ(ierr);
      ierr = CountColors(iscoloring,&adctx->p);CHKERRQ(ierr);

      // Generate seed matrix
      Seed = myalloc2(adctx->n,adctx->p);
      ierr = GenerateSeedMatrix(iscoloring,Seed);CHKERRQ(ierr);
      ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
      if (user.adctx->sparse_view) {
        ierr = PrintMat(comm,"Seed matrix:",adctx->n,adctx->p,Seed);CHKERRQ(ierr);
      }

      // Generate recovery matrix
      Rec = myalloc2(adctx->m,adctx->p);
      ierr = GetRecoveryMatrix(Seed,JP,adctx->m,adctx->p,Rec);CHKERRQ(ierr);

      // Store results and free workspace
      adctx->Rec = Rec;
      for (i=0;i<adctx->m;i++)
        free(JP[i]);
      free(JP);
      ierr = PetscFree(u_vec);CHKERRQ(ierr);
    } else {
      adctx->p = adctx->n;
      Seed = myalloc2(adctx->n,adctx->p);
      ierr = Subidentity(adctx->n,0,Seed);CHKERRQ(ierr);
    }
    adctx->Seed = Seed;

    if (adctx->zos)
      PetscPrintf(comm,"    If ||F_zos(x) - F_rhs(x)||_2/||F_rhs(x)||_2 is O(1.e-8), ADOL-C function evaluation\n      is probably correct.\n");
  }

  /* Set Jacobian */
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  if (!byhand) {
    ierr = TSSetRHSJacobian(ts,J,J,RHSJacobianAdolc,NULL);CHKERRQ(ierr);
  } else {
    ierr = TSSetRHSJacobian(ts,J,J,RHSJacobianByHand,NULL);CHKERRQ(ierr);
  }

  ftime = 1.0;
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(da,u,&user);CHKERRQ(ierr);
  dt   = .01;
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,u);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space and call destructors for active fields.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if (!adctx->no_an) {
    if (adctx->sparse)
      myfree2(Rec);
    myfree2(Seed);
    f_a += gys;
    u_a += gys;
    delete[] f_a;
    delete[] u_a;
    delete[] f_c;
    delete[] u_c;
  }
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFree(adctx);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

    test:
      args: -ts_max_steps 5 -ts_monitor 

    test:
      suffix: 2
      args: -ts_max_steps 5 -ts_monitor

    test:
      suffix: 3
      args: -ts_max_steps 5 -snes_fd_color -ts_monitor

TEST*/

