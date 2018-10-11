static char help[] = "Demonstrates automatic Jacobian diagonal generation using ADOL-C for a time-dependent PDE in 2d, solved using implicit timestepping.\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>            // Includes ADOL-C
#include <adolc/adolc_sparse.h>     // Includes ADOL-C sparse drivers
#include "utils/jacobian.cpp"

int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x,r,xdot;            /* solution, residual, derivative */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;              /* Application context */
  AdolcCtx       *adctx;
  PetscInt       gys,gxm,gym,i,dofs = 2,ctrl[3] = {0,0,0};
  AField         **u_a = NULL,**f_a = NULL,**udot_a = NULL,*u_c = NULL,*f_c = NULL,*udot_c = NULL;
  PetscScalar    **Seed = NULL,**Rec = NULL,*rec = NULL,*u_vec;
  unsigned int   **JP = NULL;
  ISColoring     iscoloring;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscMalloc1(1,&adctx);CHKERRQ(ierr);
  adctx->no_an = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse",&adctx->sparse,NULL);CHKERRQ(ierr);
  PetscFunctionBeginUser;
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
  adctx->m = dofs*gxm*gym;
  adctx->n = dofs*gxm*gym;

  // Create contiguous 1-arrays of AFields
  u_c = new AField[gxm*gym];
  f_c = new AField[gxm*gym];
  udot_c = new AField[gxm*gym];

  // Corresponding 2-arrays of AFields
  u_a = new AField*[gym];
  f_a = new AField*[gym];
  udot_a = new AField*[gym];

  // Align indices between array types and endow ghost points
  ierr = AFieldGiveGhostPoints2d(da,u_c,&u_a);CHKERRQ(ierr);
  ierr = AFieldGiveGhostPoints2d(da,f_c,&f_a);CHKERRQ(ierr);
  ierr = AFieldGiveGhostPoints2d(da,udot_c,&udot_a);CHKERRQ(ierr);

  // Store active variables in context
  appctx.u_a = u_a;
  appctx.f_a = f_a;
  appctx.udot_a = udot_a;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Trace function just once
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = IFunction(ts,1.,x,xdot,r,&appctx);CHKERRQ(ierr);
  ierr = IFunction2(ts,1.,x,xdot,r,&appctx);CHKERRQ(ierr);

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
    if (adctx->sparse_view) {
      ierr = PrintSparsity(MPI_COMM_WORLD,adctx->m,JP);CHKERRQ(ierr);
    }

    // Extract colouring
    ierr = GetColoring(da,&iscoloring);CHKERRQ(ierr);
    ierr = CountColors(iscoloring,&adctx->p);CHKERRQ(ierr);

    // Generate seed matrix and recovery vector
    Seed = myalloc2(adctx->n,adctx->p);
    ierr = PetscMalloc1(adctx->m,&rec);CHKERRQ(ierr);
    ierr = GenerateSeedMatrixPlusRecovery(iscoloring,Seed,rec);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    if (adctx->sparse_view) {
      ierr = PrintMat(MPI_COMM_WORLD,"Seed matrix:",adctx->n,adctx->p,Seed);CHKERRQ(ierr);
    }

    // Generate recovery matrix
    Rec = myalloc2(adctx->m,adctx->p);
    ierr = GetRecoveryMatrix(Seed,JP,adctx->m,adctx->p,Rec);CHKERRQ(ierr);

    // Store results and free workspace
    adctx->Rec = Rec;
    adctx->rec = rec;
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set Jacobian. In this case, IJacobian simply acts to pass context
     information to the matrix-free Jacobian vector product.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetIJacobian(ts,NULL,NULL,IJacobianAdolcDiagonal,&appctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = InitialConditions(da,x);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

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
  ierr = VecDestroy(&xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if (adctx->sparse) {
    ierr = PetscFree(rec);CHKERRQ(ierr);
    myfree2(Rec);
    myfree2(Seed);
  }
  udot_a += gys;
  f_a += gys;
  u_a += gys;
  delete[] udot_a;
  delete[] f_a;
  delete[] u_a;
  delete[] udot_c;
  delete[] f_c;
  delete[] u_c;
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFree(adctx);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
