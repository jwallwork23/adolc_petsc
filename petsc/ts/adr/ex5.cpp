static char help[] = "Demonstrates Pattern Formation with Reaction-Diffusion Equations.\n";

/*
      See ex5.c for details on the equations.
      This code applies the operator overloading automatic differentiation techniques provided by ADOL-C to automatically generate Jacobians for nonlinear partial differential equations (PDEs). Whilst this is unnecessary for equations such as these, where the Jacobian can be derived quite easily, automatic Jacobian generation would be greatly beneficial for more complex PDEs.
      Handcoded Jacobians are included here for comparison.
*/

/*
      Helpful runtime monitor options:
           -ts_monitor_draw_solution
           -draw_save -draw_save_movie

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

#define tag 1

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} AField;

typedef struct {
  PetscReal D1,D2,gamma,kappa;
  PetscBool zos,zos_view,no_an,sparse,sparse_view;
  AField    **u_a,**f_a;
} AppCtx;

/* (Slightly modified) functions included in original code of ex5.c */
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*),InitialConditions(DM,Vec);
extern PetscErrorCode RHSLocalPassive(DM da,Field **f,Field **u,void *ptr);
extern PetscErrorCode RHSJacobianByHand(TS,PetscReal,Vec,Mat,Mat,void*);

/* Problem specific functions for the purpose of automatic Jacobian computation */
extern PetscErrorCode RHSJacobianADOLC(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSLocalActive(DM da,Field **f,Field **u,void *ptr);

/* Utility functions for automatic Jacobian computation */
extern PetscErrorCode AFieldCreate2d(DM da,AField *cgs,AField **a2d);
extern PetscErrorCode AFieldGiveGhostPoints2d(DM da,AField *cgs,AField **a2d[]);
extern PetscErrorCode AFieldDestroy2d(DM da,AField *cgs[],AField **a2d[]);
extern PetscErrorCode PrintMat(MPI_Comm comm,const char* name,PetscInt m,PetscInt n,PetscScalar **M);
extern PetscErrorCode PrintSparsity(MPI_Comm comm,PetscInt m,unsigned int **JP);
extern PetscErrorCode GetColoring(MPI_Comm comm,PetscInt m,PetscInt n,unsigned int **JP,PetscInt *p,ISColoring *iscoloring);
extern PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring,PetscInt n,PetscInt p,PetscScalar **Seed);
extern PetscErrorCode GetRecoveryMatrix(PetscScalar **Seed,unsigned int **JP,PetscInt m,PetscInt p,PetscScalar **Rec);
extern PetscErrorCode RecoverJacobian(Mat J,PetscInt m,PetscInt p,PetscScalar **Rec,PetscScalar **Jcomp);
extern PetscErrorCode TestZOS2d(DM da,Field **f,Field **u,void *ctx);

int main(int argc,char **argv)
{
  TS             ts;                    /* ODE integrator */
  Vec            x;                     /* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  PetscInt       gxs,gys,gxm,gym;
  AField         **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;
  PetscBool      byhand = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  appctx.zos = PETSC_FALSE;appctx.zos_view = PETSC_FALSE;appctx.no_an = PETSC_FALSE;appctx.sparse = PETSC_FALSE;appctx.sparse_view = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos",&appctx.zos,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos_view",&appctx.zos_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse",&appctx.sparse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse_view",&appctx.sparse_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&appctx.no_an,NULL);CHKERRQ(ierr);
  appctx.D1     = 8.0e-5;
  appctx.D2     = 4.0e-5;
  appctx.gamma  = .024;
  appctx.kappa  = .06;

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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Allocate memory for (local) active fields (called AFields) and store 
     references in the application context. The AFields are reused at
     each active section, so need only be created once.

     NOTE: Memory for ADOL-C active variables (such as adouble and AField)
           cannot be allocated using PetscMalloc, as this does not call the
           relevant class constructor. Instead, we use the C++ keyword `new`.

           It is also important to deconstruct and free memory appropriately.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!appctx.no_an) {

    ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

    // Create contiguous 1-arrays of AFields
    u_c = new AField[gxm*gym];
    f_c = new AField[gxm*gym];

    // Corresponding 2-arrays of AFields
    u_a = new AField*[gym];
    f_a = new AField*[gym];
/*
    ierr = AFieldCreate2d(da,u_c,u_a);CHKERRQ(ierr);
    ierr = AFieldCreate2d(da,f_c,f_a);CHKERRQ(ierr);
*/
    // Align indices between array types and endow ghost points
    ierr = AFieldGiveGhostPoints2d(da,u_c,&u_a);CHKERRQ(ierr);
    ierr = AFieldGiveGhostPoints2d(da,f_c,&f_a);CHKERRQ(ierr);

    // Store active variables in context
    appctx.u_a = u_a;
    appctx.f_a = f_a;
  }

  if (appctx.zos) {
    PetscPrintf(MPI_COMM_WORLD,"    If ||F_zos(x) - F_rhs(x)||_2/||F_rhs(x)||_2 is O(1.e-8), ADOL-C function evaluation\n      is probably correct.\n");
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);
  if (!byhand) {
    ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobianADOLC,&appctx);CHKERRQ(ierr);
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
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if (!appctx.no_an) {
/*
    ierr = AFieldDestroy2d(da,f_c,f_a);CHKERRQ(ierr);
    ierr = AFieldDestroy2d(da,u_c,u_a);CHKERRQ(ierr);
*/
    f_a += gys;
    u_a += gys;
    delete[] f_a;
    delete[] u_a;
    delete[] f_c;
    delete[] u_c;
  }
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode RHSLocalActive(DM da,Field **f,Field **u,void *ptr)
{
  PetscErrorCode  ierr;
  AppCtx          *appctx = (AppCtx*)ptr;
  PetscInt        i,j,xs,ys,xm,ym,gxs,gys,gxm,gym,Mx,My;
  PetscReal       hx,hy,sx,sy;
  AField          **f_a = appctx->f_a,**u_a = appctx->u_a;
  adouble         uc,uxx,uyy,vc,vxx,vyy;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)(Mx);sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My);sy = 1.0/(hy*hy);

  trace_on(tag);  // ----------------------------------------------- Start of active section

  /*
    Mark independence

    NOTE: Ghost points are marked as independent, in place of the points they represent on
          other processors / on other boundaries.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_a[j][i].u <<= u[j][i].u;
      u_a[j][i].v <<= u[j][i].v;
    }
  }

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {

      // Compute function over the locally owned part of the grid
      uc          = u_a[j][i].u;
      uxx         = (-2.0*uc + u_a[j][i-1].u + u_a[j][i+1].u)*sx;
      uyy         = (-2.0*uc + u_a[j-1][i].u + u_a[j+1][i].u)*sy;
      vc          = u_a[j][i].v;
      vxx         = (-2.0*vc + u_a[j][i-1].v + u_a[j][i+1].v)*sx;
      vyy         = (-2.0*vc + u_a[j-1][i].v + u_a[j+1][i].v)*sy;
      f_a[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);
      f_a[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc;
    }
  }

  /*
    Mark dependence

    NOTE: Ghost points are marked as dependent in order to vastly simplify index notation
          during Jacobian assembly.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      f_a[j][i].u >>= f[j][i].u;
      f_a[j][i].v >>= f[j][i].v;
    }
  }
  trace_off();  // ----------------------------------------------- End of active section

  PetscFunctionReturn(0);
}

PetscErrorCode RHSLocalPassive(DM da,Field **f,Field **u,void *ptr)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)(Mx);sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My);sy = 1.0/(hy*hy);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;
      vc        = u[j][i].v;
      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
      f[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);
      f[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc;
    }
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   RHSFunction - Evaluates nonlinear function, F(x).

                 If the -no_annotations option is not invoked then
                 annotations are made for ADOL-C automatic
                 differentiation using an `AField` struct.

   Input Parameters:
.  ts - the TS context
.  X - input vector
.  ptr - optional user-defined context, as set by TSSetRHSFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode RHSFunction(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       xm,ym;
  Field          **u,**f;
  Vec            localU,localF;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localF);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr); // NOTE (1): See (2) below
  ierr = DMGlobalToLocalBegin(da,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,F,INSERT_VALUES,localF);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localF,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,NULL,NULL,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  if (!appctx->no_an) {
    ierr = RHSLocalActive(da,f,u,appctx);CHKERRQ(ierr);

    /* Test zeroth order scalar evaluation in ADOL-C gives the same result */
    if (appctx->zos) {
      ierr = TestZOS2d(da,f,u,appctx);CHKERRQ(ierr); // FIXME: Why does this give nonzero?
    }
  } else {
    ierr = RHSLocalPassive(da,f,u,appctx);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);

  /*
     Gather global vector, using the 2-step process
        DMLocalToGlobalBegin(),DMLocalToGlobalEnd().

     NOTE (2): We need to use ADD_VALUES if boundaries are not of type DM_BOUNDARY_NONE or 
               DM_BOUNDARY_GHOSTED, meaning we should also zero F before addition (see (1) above).
  */
  ierr = DMLocalToGlobalBegin(da,localF,ADD_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,localF,ADD_VALUES,F);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,localF,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localF);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode InitialConditions(DM da,Vec U)
{
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  Field          **u;
  PetscReal      hx,hy,x,y;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.5/(PetscReal)(Mx);
  hy = 2.5/(PetscReal)(My);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      if ((1.0 <= x) && (x <= 1.5) && (1.0 <= y) && (y <= 1.5)) u[j][i].v = .25*PetscPowReal(PetscSinReal(4.0*PETSC_PI*x),2.0)*PetscPowReal(PetscSinReal(4.0*PETSC_PI*y),2.0);
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0*u[j][i].v;
    }
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobianADOLC(TS ts,PetscReal t,Vec U,Mat A,Mat BB,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,k = 0,xm,ym,gxs,gys,gxm,gym,m,n,dofs = 2;
  PetscScalar    *u_vec,**J = NULL;
  Field          **u;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);

  /* Get local and ghosted grid boundaries */
  ierr = DMDAGetCorners(da,NULL,NULL,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  /* Convert array of structs to a 1-array, so this can be read by ADOL-C */
  m = dofs*gxm*gym;    // Number of dependent variables
  n = m;               // Number of independent variables
  ierr = PetscMalloc1(n,&u_vec);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_vec[k++] = u[j][i].u;
      u_vec[k++] = u[j][i].v;
    }
  }
  k = 0;

  /*
    Calculate Jacobian using ADOL-C
  */
  if (appctx->sparse) {
    ierr = PetscPrintf(MPI_COMM_WORLD,"Exiting. Sparse driver not yet complete.\n");CHKERRQ(ierr);
    exit(0);

  } else {

    J = myalloc2(m,n);
    jacobian(tag,m,n,u_vec,J);
    ierr = PetscFree(u_vec);CHKERRQ(ierr);

    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
    myfree2(J);
  }

  /*
     Restore vectors
  */
  ierr = PetscLogFlops(19*xm*ym);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);

  /*
    Assemble local matrix
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobianByHand(TS ts,PetscReal t,Vec U,Mat A,Mat BB,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,vc;
  Field          **u;
  Vec            localU;
  MatStencil     stencil[6],rowstencil;
  PetscScalar    entries[6];

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.50/(PetscReal)(Mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My); sy = 1.0/(hy*hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  stencil[0].k = 0;
  stencil[1].k = 0;
  stencil[2].k = 0;
  stencil[3].k = 0;
  stencil[4].k = 0;
  stencil[5].k = 0;
  rowstencil.k = 0;

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {

    stencil[0].j = j-1;
    stencil[1].j = j+1;
    stencil[2].j = j;
    stencil[3].j = j;
    stencil[4].j = j;
    stencil[5].j = j;
    rowstencil.k = 0; rowstencil.j = j;
    for (i=xs; i<xs+xm; i++) {
      uc = u[j][i].u;
      vc = u[j][i].v;

      /*      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;

      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
       f[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);*/

      stencil[0].i = i; stencil[0].c = 0; entries[0] = appctx->D1*sy;
      stencil[1].i = i; stencil[1].c = 0; entries[1] = appctx->D1*sy;
      stencil[2].i = i-1; stencil[2].c = 0; entries[2] = appctx->D1*sx;
      stencil[3].i = i+1; stencil[3].c = 0; entries[3] = appctx->D1*sx;
      stencil[4].i = i; stencil[4].c = 0; entries[4] = -2.0*appctx->D1*(sx + sy) - vc*vc - appctx->gamma;
      stencil[5].i = i; stencil[5].c = 1; entries[5] = -2.0*uc*vc;
      rowstencil.i = i; rowstencil.c = 0;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);

      stencil[0].c = 1; entries[0] = appctx->D2*sy;
      stencil[1].c = 1; entries[1] = appctx->D2*sy;
      stencil[2].c = 1; entries[2] = appctx->D2*sx;
      stencil[3].c = 1; entries[3] = appctx->D2*sx;
      stencil[4].c = 1; entries[4] = -2.0*appctx->D2*(sx + sy) + 2.0*uc*vc - appctx->gamma - appctx->kappa;
      stencil[5].c = 0; entries[5] = vc*vc;
      rowstencil.c = 1;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      /* f[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc; */
    }
  }

  /*
     Restore vectors
  */
  ierr = PetscLogFlops(19*xm*ym);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Set up AField, including ghost points.

  FIXME: How to do this properly?
*/
PetscErrorCode AFieldCreate2d(DM da,AField *cgs,AField **a2d)
{
  PetscErrorCode ierr;
  PetscInt       gxs,gys,gxm,gym;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  cgs = new AField[gxm*gym];	// Contiguous 1-arrays of AFields
  a2d = new AField*[gym];	// Corresponding 2-arrays of AFields
  PetscFunctionReturn(0);
}

/*
  Shift indices in AField to endow it with ghost points.
*/
PetscErrorCode AFieldGiveGhostPoints2d(DM da,AField *cgs,AField **a2d[])
{
  PetscErrorCode ierr;
  PetscInt       gxs,gys,gxm,gym,j;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  for (j=0; j<gym; j++) {
    (*a2d)[j] = cgs + j*gxm - gxs;
  }
  *a2d -= gys;
  PetscFunctionReturn(0);
}

/*
  Destroy AField.

  FIXME: How to do this properly?
*/
PetscErrorCode AFieldDestroy2d(DM da,AField *cgs[],AField **a2d[])
{
  PetscErrorCode ierr;
  PetscInt       gys;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,NULL,&gys,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  *a2d += gys;
  delete[] a2d;
  delete[] cgs;

  PetscFunctionReturn(0);
}

/*
  Print matrices involved in sparse computations.
*/
PetscErrorCode PrintMat(MPI_Comm comm,const char* name,PetscInt m,PetscInt n,PetscScalar **M)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"%s \n",name);CHKERRQ(ierr);
  for(i=0; i<m ;i++) {
    ierr = PetscPrintf(comm,"\n %d: ",i);CHKERRQ(ierr);
    for(j=0; j<n ;j++)
      ierr = PetscPrintf(comm," %10.4f ", M[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PrintSparsity(MPI_Comm comm,PetscInt m,unsigned int **JP)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"Sparsity pattern:\n");CHKERRQ(ierr);
  for(i=0; i<m ;i++) {
    ierr = PetscPrintf(comm,"\n %2d: ",i);CHKERRQ(ierr);
    for(j=1; j<= (PetscInt) JP[i][0] ;j++)
      ierr = PetscPrintf(comm," %2d ", JP[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GetColoring(MPI_Comm comm,PetscInt m,PetscInt n,unsigned int **JP,PetscInt *p,ISColoring *iscoloring)
{
  PetscErrorCode ierr;
  MatColoring    coloring;
  PetscScalar    one = 1.;
  Mat            S;
  PetscInt       i,j,k;

  PetscFunctionBegin;

  // Create Jacobian sparsity pattern object, assembling with preallocated nonzeros as ones
  ierr = MatCreate(comm,&S);CHKERRQ(ierr);
  ierr = MatSetSizes(S,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(S);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  *p = 0;
  for (i=0;i<m;i++) {
    if ((PetscInt) JP[i][0] > *p)
      *p = (PetscInt) JP[i][0];
    for (j=1;j<= (PetscInt) JP[i][0];j++) {
      k = JP[i][j];
      ierr = MatSetValues(S,1,&i,1,&k,&one,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // Extract colouring
  ierr = MatColoringCreate(S,&coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);      // 'Smallest last' default
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring,iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring,PetscInt n,PetscInt p,PetscScalar **Seed)
{
  PetscErrorCode ierr;
  IS             *isp,is;
  PetscInt       nis,size,i,j;
  const PetscInt *indices;

  PetscFunctionBegin;

  ierr = ISColoringGetIS(iscoloring,&nis,&isp);CHKERRQ(ierr);
  for (i=0;i<p;i++) {
    is = *(isp+i);
    ierr = ISGetLocalSize(is,&size);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
    for (j=0;j<size;j++) {
      Seed[indices[j]][i] = 1.;
    }
    ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&isp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GetRecoveryMatrix(PetscScalar **Seed,unsigned int **JP,PetscInt m,PetscInt p,PetscScalar **Rec)
{
  PetscInt i,j,k,colour;

  PetscFunctionBegin;
  for (i=0;i<m;i++) {
    for (colour=0;colour<p;colour++) {
      Rec[i][colour] = -1.;
      for (k=1;k<=(PetscInt) JP[i][0];k++) {
        j = (PetscInt) JP[i][k];
        if (Seed[j][colour] == 1.) {
          Rec[i][colour] = j;
          break;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RecoverJacobian(Mat J,PetscInt m,PetscInt p,PetscScalar **Rec,PetscScalar **Jcomp)
{
  PetscErrorCode ierr;
  PetscInt       i,j,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0;colour<p;colour++) {
      j = (PetscInt) Rec[i][colour];
      if (j != -1)
        ierr = MatSetValuesLocal(J,1,&i,1,&j,&Jcomp[i][colour],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TestZOS2d(DM da,Field **f,Field **u,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       m,n,gxs,gys,gxm,gym,i,j,k = 0,dofs = 2;
  PetscScalar    diff = 0,norm = 0,*u_vec,*fz;
  MPI_Comm       comm = MPI_COMM_WORLD;

  PetscFunctionBegin;

  /* Get extent of region owned by processor */
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  m = dofs*gxm*gym;
  n = m;

  /* Convert to a 1-array */
  ierr = PetscMalloc1(n,&u_vec);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++)
      u_vec[k++] = u[j][i].u;
      u_vec[k++] = u[j][i].v;
  }
  k = 0;

  /* Zero order scalar evaluation vs. calling RHS function */
  ierr = PetscMalloc1(m,&fz);CHKERRQ(ierr);
  zos_forward(tag,m,n,0,u_vec,fz);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      if ((appctx->zos_view) && (fabs(f[j][i].u) > 1.e-16) && (fabs(fz[k]) > 1.e-16))
        PetscPrintf(comm,"(%2d,%2d): F_rhs = %+.4e, F_zos = %+.4e\n",j,i,f[j][i].u,fz[k]);
      diff += (f[j][i].u-fz[k])*(f[j][i].u-fz[k]);k++;
      norm += f[j][i].u*f[j][i].u;
      if ((appctx->zos_view) && (fabs(f[j][i].v) > 1.e-16) && (fabs(fz[k]) > 1.e-16))
        PetscPrintf(comm,"(%2d,%2d): F_rhs = %+.4e, F_zos = %+.4e\n",j,i,f[j][i].v,fz[k]);
      diff += (f[j][i].v-fz[k])*(f[j][i].v-fz[k]);k++;
      norm += f[j][i].v*f[j][i].v;
    }
  }
  norm = sqrt(diff/norm);
  PetscPrintf(comm,"    ----- Testing Zero Order evaluation -----\n");
  PetscPrintf(comm,"    ||F_zos(x) - F_rhs(x)||_2/||F_rhs(x)||_2 = %.4e\n",norm);
  ierr = PetscFree(fz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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
