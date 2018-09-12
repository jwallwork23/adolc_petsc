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
           -jacobian_by_hand - use a hand-coded Jacobian
           -no_annotations   - do not annotate using ADOL-C
           -sparse           - generate Jacobian using ADOL-C sparse Jacobian driver
           -sparse_row       - use row compression, rather than column compression

      Helpful ADOL-C debugging options:
           -adolc_test_zos      - test Zero Order Scalar evaluation
           -adolc_test_zos_view - view each nonzero contribution to check validity

      Helpful runtime monitor options for debugging:
           -da_grid_x 12 -da_grid_y 12 -ts_max_steps 1 -snes_test_jacobian
           -da_grid_x 12 -da_grid_y 12 -ts_max_steps 1 -snes_test_jacobian -snes_test_jacobian_view

      Helpful runtime linear solver options:
           -pc_type mg -pc_mg_galerkin pmat -da_refine 1 -snes_monitor -ksp_monitor -ts_view  (note that these Jacobians are so well-conditioned multigrid may not be the best solver)

      Point your browser to localhost:8080 to monitor the simulation
           ./ex5  -ts_view_pre saws  -stack_view saws -draw_save -draw_save_single_file -x_virtual -ts_monitor_draw_solution -saws_root .
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

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} AField;

typedef struct {
  PetscReal D1,D2,gamma,kappa;
  PetscBool zos,zos_view,no_an,sparse,sparse_row;
  PetscInt  Mx,My;
  AField    **u_a,**f_a;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*),InitialConditions(DM,Vec);
extern PetscErrorCode RHSJacobianADOLC(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSJacobianByHand(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSLocalActive(DM da,Field **f,Field **u,void *ptr);
extern PetscErrorCode RHSLocalPassive(DM da,Field **f,Field **u,void *ptr);

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
  Insert ghost point values into AField.
*/
PetscErrorCode AFieldInsertGhostValues2d(DM da,Field **u,AField **u_a)
{
  PetscErrorCode  ierr;
  PetscInt        i,j,xs,ys,xm,ym,gxs,gys,gxm,gym,lower,upper;
  DMDAStencilType st;
  DMBoundaryType  xbc,ybc;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,&xbc,&ybc,PETSC_IGNORE,&st);CHKERRQ(ierr);

  // Ghost points need to be present, even if unused, as with DM_BOUNDARY_GHOSTED.
  if ((xbc != DM_BOUNDARY_NONE) && (ybc != DM_BOUNDARY_NONE)) {
    lower = ys;upper = ys+ym;
    if (st == DMDA_STENCIL_BOX) {
      lower += gys;upper -= gys;
    }
    for (j=lower; j<upper; j++) {
      for (i=0; i<2; i++) {
        u_a[j][gxs+i*(gxm-1)].u = u[j][gxs+i*(gxm-1)].u;
        u_a[j][gxs+i*(gxm-1)].v = u[j][gxs+i*(gxm-1)].v;
      }
    }
    lower = xs;upper = xs+xm;
    if (st == DMDA_STENCIL_BOX) {
      lower += gxs;upper -= gxs;
    }
    for (i=lower; i<upper; i++) {
      for (j=0; j<2; j++) {
        u_a[gys+j*(gym-1)][i].u = u[gys+j*(gym-1)][i].u;
        u_a[gys+j*(gym-1)][i].v = u[gys+j*(gym-1)][i].v;
      }
    }
  } else {
    SETERRQ(PETSC_COMM_SELF,1,"Ghost points required on boundary.");
  }
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

int main(int argc,char **argv)
{
  TS             ts;                    /* ODE integrator */
  Vec            x;                     /* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  AField         **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;
  PetscInt       gxs,gys,gxm,gym;
  PetscBool      byhand = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  appctx.zos = PETSC_FALSE;appctx.zos_view = PETSC_FALSE;appctx.no_an = PETSC_FALSE;appctx.sparse = PETSC_FALSE;appctx.sparse_row = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos",&appctx.zos,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos_view",&appctx.zos_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&appctx.no_an,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-sparse",&appctx.sparse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-sparse_row",&appctx.sparse_row,NULL);CHKERRQ(ierr);
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
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&appctx.Mx,&appctx.My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

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
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Call destructors for active fields, freeing associated memory in the
     process.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
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

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode RHSLocalActive(DM da,Field **f,Field **u,void *ptr)
{
  PetscErrorCode  ierr;
  AppCtx          *appctx = (AppCtx*)ptr;
  PetscInt        i,j,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal       hx,hy,sx,sy;
  AField          **f_a = appctx->f_a,**u_a = appctx->u_a;
  adouble         uc,uxx,uyy,vc,vxx,vyy;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)(appctx->Mx);sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(appctx->My);sy = 1.0/(hy*hy);

  trace_on(1);  // ----------------------------------------------- Start of active section

  /*
    Mark independence

    NOTE: Ghost points are marked as independent at this stage, but their contributions to
          the Jacobian will be added to the corresponding rows on other processes, meaning
          the Jacobian remains square.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_a[j][i].u <<= u[j][i].u;u_a[j][i].v <<= u[j][i].v;
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

      // Mark dependence
      f_a[j][i].u >>= f[j][i].u;f_a[j][i].v >>= f[j][i].v;
    }
  }
  trace_off();  // ----------------------------------------------- End of active section

  PetscFunctionReturn(0);
}

PetscErrorCode RHSLocalPassive(DM da,Field **f,Field **u,void *ptr)
{
  PetscErrorCode ierr;
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)(appctx->Mx);sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(appctx->My);sy = 1.0/(hy*hy);
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
  PetscInt       xs,ys,xm,ym;
  Field          **u,**f;
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

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  if (!appctx->no_an) {
    ierr = RHSLocalActive(da,f,u,appctx);CHKERRQ(ierr);
  } else {
    ierr = RHSLocalPassive(da,f,u,appctx);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
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
  PetscInt       i,j,k = 0,w_ghost = 0,wo_ghost = 0,xs,ys,xm,ym,gxs,gys,gxm,gym,L,G;
  PetscInt       nnz,options[4] = {0,0,0,0},loc,d,dofs = 2;
  unsigned int   *rind = NULL,*cind = NULL;
  PetscScalar    *u_vec,**J = NULL,norm=0.,diff=0.,*fz,*values = NULL;
  Field          **u,**frhs;
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
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  /* Convert array of structs to a 1-array, so this can be read by ADOL-C */
  L = dofs*xm*ym;G = dofs*gxm*gym;
  ierr = PetscMalloc1(G,&u_vec);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_vec[k++] = u[j][i].u;u_vec[k++] = u[j][i].v;
    }
  }

  /* Test zeroth order scalar evaluation in ADOL-C gives the same result as calling RHSLocalPassive */
  if (appctx->zos) {
    k = 0;
    ierr = PetscMalloc1(L,&fz);CHKERRQ(ierr);
    zos_forward(1,L,G,0,u_vec,fz);
    frhs = new Field*[ym];
    for (j=ys; j<ys+ym; j++)
      frhs[j] = new Field[xm];
    RHSLocalPassive(da,frhs,u,appctx);

    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        if (appctx->zos_view) {
          if ((fabs(frhs[j][i].u) > 1.e-16) && (fabs(fz[k]) > 1.e-16)) {
            PetscPrintf(MPI_COMM_WORLD,"F_rhs[%2d,%2d,u] = %+.4e, ",j,i,frhs[j][i].u);
            PetscPrintf(MPI_COMM_WORLD,"F_zos[%2d,%2d,u] = %+.4e\n",j,i,fz[k++]);
          }
          if ((fabs(frhs[j][i].v) > 1.e-16) && (fabs(fz[k]) > 1.e-16)) {
            PetscPrintf(MPI_COMM_WORLD,"F_rhs[%2d,%2d,v] = %+.4e, ",j,i,frhs[j][i].v);
            PetscPrintf(MPI_COMM_WORLD,"F_zos[%2d,%2d,v] = %+.4e\n",j,i,fz[k--]);
          }
        }
        diff += (frhs[j][i].u-fz[k])*(frhs[j][i].u-fz[k]);k++;
        diff += (frhs[j][i].v-fz[k])*(frhs[j][i].v-fz[k]);k++;
        norm += frhs[j][i].u*frhs[j][i].u + frhs[j][i].v*frhs[j][i].v;
      }
    }
    ierr = PetscFree(fz);CHKERRQ(ierr);
    for (j=ys; j<ys+ym; j++)
      delete[] frhs[j];
    delete[] frhs;
    PetscPrintf(MPI_COMM_WORLD,"    ----- Testing Zero Order evaluation -----\n");
    PetscPrintf(MPI_COMM_WORLD,"    ||F_zos(x) - F_rhs(x)||_2/||F_rhs(x)||_2 = %.4e\n",sqrt(diff/norm));
  }

  /*
    Calculate Jacobian using ADOL-C
  */
  if (appctx->sparse) {  // TODO: First need to link properly with ColPack

    if (appctx->sparse_row) {
      options[3] = 1;
    }

    sparse_jac(1,L,G,0,u_vec,&nnz,&rind,&cind,&values,options);	// TODO: Consider separate drivers
    for (k=0; k<nnz; k++){
      j = rind[k];i = cind[k]; // TODO: These need adjustment for ghost points, as below.
      ierr = MatSetValues(A,1,&j,1,&i,&values[k],INSERT_VALUES);CHKERRQ(ierr);
    }
    free(rind);rind = NULL;
    free(cind);cind = NULL;
    free(values);values = NULL;

  } else {

    J = myalloc2(L,G);
    jacobian(1,L,G,u_vec,J);
    ierr = PetscFree(u_vec);CHKERRQ(ierr);

    /* Insert entries one-by-one. TODO: better to insert row-by-row, similarly as with the stencil */
    for (k=0; k<L; k++) {
      for (j=gys; j<gys+gym; j++) {
        for (i=gxs; i<gxs+gxm; i++) {
          for (d=0; d<dofs; d++) {

            // CASE 1: ghost point below local region
            if (j < ys) {

              // Bottom boundary
              if ((j < 0) && (i >= 0) && (i < appctx->Mx))
                loc = d+dofs*(i+appctx->Mx*(appctx->My+j));
              else
                loc = d+dofs*(i+xm*(ym+j));	// TODO: Test this

            // CASE 2: ghost point above local region
            } else if (j >= ym) {

              // Top boundary
              if ((j >= appctx->My) && (i >= 0) && (i < appctx->Mx))
                loc = d+dofs*i;
              else
                loc = d+dofs*i;			// TODO: Test this

            // CASE 3: ghost point left of local region
            } else if (i < xs) {

              // Left boundary
              if ((i < 0) && (j >= 0) && (j < appctx->My))
                loc = wo_ghost+dofs*(appctx->Mx+i)+d;
              else
                loc = wo_ghost+dofs*(xm+i)+d;	// TODO: Test this

            // CASE 4: ghost point right of local region
            } else if (i >= xm) {

              // Right boundary
              if ((i >= appctx->Mx) && (j >= 0) && (j < appctx->My))
                loc = wo_ghost-dofs*(2*appctx->Mx-i)+d;
              else
                loc = wo_ghost-dofs*(2*xm-i)-d;	// TODO: Test this

            // CASE 5: Interior points of local region
            } else {
              loc = wo_ghost;
              wo_ghost++;
            }
            if (fabs(J[k][w_ghost])!=0.)
              ierr = MatSetValues(A,1,&k,1,&loc,&J[k][w_ghost],INSERT_VALUES);CHKERRQ(ierr);
            w_ghost++;
          }
        }
      }
      w_ghost = 0;wo_ghost = 0;
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

    NOTE (from Vec ex2): Each processor can contribute any vector entries, regardless of which
         processor "owns" them; any nonlocal contributions will be transferred to the appropriate
         processor during the assembly process.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
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
