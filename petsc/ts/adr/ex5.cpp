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
           -analytic       - use a hand-coded Jacobian
           -no_annotations - do not annotate using ADOL-C
           -sparse         - generate Jacobian using ADOL-C sparse Jacobian driver

      Helpful ADOL-C related options:
           -adolc_test_zos (test Zero Order Scalar evaluation)

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
#include <adolc/sparse/sparsedrivers.h>
#include "utils.c"		// For modular arithmetic and coordinate mappings

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} aField;

typedef struct {
  PetscReal D1,D2,gamma,kappa;
  PetscBool zos,no_an,sparse;
  PetscInt  Mx,My;
  aField    **u_a,**f_a;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*),InitialConditions(DM,Vec);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSJacobianByHand(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSLocalActive(Field **f,Field **u,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,void *ptr);
extern PetscErrorCode RHSLocalPassive(Field **f,Field **u,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,void *ptr);

PetscErrorCode ShiftIndices(adouble *arr,PetscInt ym,PetscInt xm,PetscInt ys,PetscInt xs,adouble **a[])
{
  PetscInt       j;
  PetscFunctionBegin;
  for (j=0; j<ym; j++) (*a)[j] = arr + j*xm - xs;
  *a -= ys;
  PetscFunctionReturn(0);
}


int main(int argc,char **argv)
{
  TS             ts;                    /* ODE integrator */
  Vec            x;                     /* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  aField         **u_a=NULL,**f_a=NULL,*u_c=NULL,*f_c=NULL;
//  aField         **u_a=NULL,**f_a=NULL;
//  adouble        *u_c=NULL,*f_c=NULL;
  PetscInt       gxs,gys,gxm,gym,j,dof;
  PetscBool      analytic=PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  appctx.zos = PETSC_FALSE;appctx.no_an = PETSC_FALSE;appctx.sparse = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos",&appctx.zos,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-analytic",&analytic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&appctx.no_an,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-sparse",&appctx.sparse,NULL);CHKERRQ(ierr);
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
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&appctx.Mx,&appctx.My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,&dof,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Allocate memory for active fields and store references in context 

     NOTE: Memory for ADOL-C active variables (such as adouble and aField)
           cannot be allocated using PetscMalloc, as this does not call the
           relevant class constructor. Instead, we use the C++ keyword `new`.

           It is also important to deconstruct and free memory appropriately.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!appctx.no_an) {
    /*
       Create active field structures and endow with ghost points.

       TODO: Enforce periodic BC.
    */

    ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

    // Create contiguous 1-arrays of aFields
    u_c = new aField[gxm*gym];
    f_c = new aField[gxm*gym];
/*
    // Contiguous 1-arrays of adoubles
    u_c = new adouble[dof*gxm*gym];
    f_c = new adouble[dof*gxm*gym];
*/
    // Corresponding 2-arrays of aFields
    u_a = new aField*[gym];
    f_a = new aField*[gym];
    for (j=0; j<gym; j++) {
      u_a[j] = new aField[gxm];
      delete[] u_a[j];
      f_a[j] = new aField[gxm];
      delete[] f_a[j];
    }

    // Shift the values to endow ghost points
    for (j=0; j<gym; j++) {
      u_a[j] = u_c + j*gxm - gxs;
      f_a[j] = f_c + j*gxm - gxs;
    }
    *u_a -= gys;
    *f_a -= gys;
/*
    for (j=0; j<gym; j++) {
      u_a[j] = u_c + j*gxm;
      f_a[j] = f_c + j*gxm;
    }
*/
/*
    ierr = ShiftIndices(u_c,gym,gxm*dof,gys,gxs*dof,(adouble***)u_a);CHKERRQ(ierr);
    ierr = ShiftIndices(f_c,gym,gxm*dof,gys,gxs*dof,(adouble***)f_a);CHKERRQ(ierr);
*/
    appctx.u_a = u_a;
    appctx.f_a = f_a;

    PetscInt i,k=0;
    for (j=gys; j<gym; j++) {
      for (i=gxs; i<gxm; i++) {
/*
        std::cout << i << "," << j << " " << &u_a[j][i].u << "," << &u_c[k].u << std::endl;
        std::cout << i << "," << j << " " << &u_a[j][i].v << "," << &u_c[k].v << std::endl;
        k++;
*/
/*
        std::cout << i << "," << j << " " << &u_a[j][i].u << "," << &u_c[k++] << std::endl;
        std::cout << i << "," << j << " " << &u_a[j][i].v << "," << &u_c[k++] << std::endl;
*/
      }
    }
/*
    std::cout << "Addresses out of reach: " << std::endl;
    for (i=0;i<12;i++){
      std::cout << &u_a[-1][i].u << ", " << &u_a[-1][i].v << std::endl;
    }
*/
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
  if (!analytic) {
    ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,&appctx);CHKERRQ(ierr);
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

  if (!appctx.no_an) {

    // Call destructors / free memory
    delete[] f_a;
    delete[] u_a;

    delete[] f_c;
    delete[] u_c;
  }

  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode RHSLocalActive(Field **f,Field **u,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,void *ptr)
{
  AppCtx          *appctx = (AppCtx*)ptr;
  aField          **f_a = appctx->f_a,**u_a = appctx->u_a;
  PetscInt        i,j,sx,sy;
  PetscReal       hx,hy;
  adouble         uc,uxx,uyy,vc,vxx,vyy;

  PetscFunctionBeginUser;

  hx = 2.50/(PetscReal)(appctx->Mx);sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(appctx->My);sy = 1.0/(hy*hy);

/*
  for (j=ys-1; j<ym+2; j++) {
    for (i=xs-1; i<xm+2; i++) {
      std::cout << i << "," << j << " " << &u[j][i].u << std::endl;
      std::cout << i << "," << j << " " << &u[j][i].v << std::endl;
    }
  }
*/

  trace_on(1);  // ----------------------------------------------- Start of active section

  // Mark independence
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      u_a[j][i].u <<= u[j][i].u;u_a[j][i].v <<= u[j][i].v;
    }
  }
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {

      // Compute function over the locally owned part of the grid
      uc          = u_a[j][i].u;
      uxx         = (-2.0*uc + u_a[j][i-1].u + u_a[j][i+1].u)*sx;
      if (j == 0) {	// TODO: remove temporary special-casing
        uyy         = (-2.0*uc + u_a[modulo(j-1,appctx->My)][i].u + u_a[j+1][i].u)*sy;
      } else {
        uyy         = (-2.0*uc + u_a[j-1][i].u + u_a[j+1][i].u)*sy;
      }
      vc          = u_a[j][i].v;
      vxx         = (-2.0*vc + u_a[j][i-1].v + u_a[j][i+1].v)*sx;
      if (j == 0) {
        vyy         = (-2.0*vc + u_a[modulo(j-1,appctx->My)][i].v + u_a[j+1][i].v)*sy;
      } else {
        vyy         = (-2.0*vc + u_a[j-1][i].v + u_a[j+1][i].v)*sy;
      }
      f_a[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);
      f_a[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc;

      // Mark dependence
      f_a[j][i].u >>= f[j][i].u;f_a[j][i].v >>= f[j][i].v;
    }
  }
  trace_off();  // ----------------------------------------------- End of active section

  PetscFunctionReturn(0);
}

PetscErrorCode RHSLocalPassive(Field **f,Field **u,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,void *ptr)
{
  AppCtx        *appctx = (AppCtx*)ptr;
  PetscInt      i,j,sx,sy;
  PetscReal     hx,hy;
  PetscScalar   uc,uxx,uyy,vc,vxx,vyy;

  PetscFunctionBeginUser;
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
                 differentiation using an `aField` struct.

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
  PetscInt       xs,ys,xm,ym,N;
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

  VecGetLocalSize(localU,&N);
  std::cout << "N = " << N << std::endl;

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  if (!appctx->no_an) {
    RHSLocalActive(f,u,xs,ys,xm,ym,appctx);
  } else {
    RHSLocalPassive(f,u,xs,ys,xm,ym,appctx);
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

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat BB,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,k = 0,Mx,My,xs,ys,xm,ym,N,dofs,col[1];
  PetscScalar    *u_vec,**J,norm=0.,diff=0.;
  Field          **u;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,&dofs,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

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
     Get local grid boundaries and total degrees of freedom on this process
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
    Convert array of structs to a 1-array, so this can be read by ADOL-C
  */
  N = 2*(xs+xm)*(ys+ym);
  ierr = PetscMalloc1(N,&u_vec);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      u_vec[k++] = u[j][i].u;u_vec[k++] = u[j][i].v;
    }
  }

  /*
    Test zeroth order scalar evaluation in ADOL-C gives the same result as calling RHSLocalPassive
  */
  if (appctx->zos) {
    k = 0;
    PetscScalar  *fz;
    Field        **frhs;

    ierr = PetscMalloc1(N,&fz);CHKERRQ(ierr);
    zos_forward(1,N,N,0,u_vec,fz);

    ierr = PetscMalloc1(N,&frhs);CHKERRQ(ierr);		// FIXME: Memory is not contiguous
    for (j=ys; j<ys+ym; j++) {
      ierr = PetscMalloc1(N,&frhs[j]);CHKERRQ(ierr);
    }
    RHSLocalPassive(frhs,u,xs,ys,xm,ym,appctx);

    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        diff += (frhs[j][i].u-fz[k])*(frhs[j][i].u-fz[k]);k++;
        diff += (frhs[j][i].v-fz[k])*(frhs[j][i].v-fz[k]);k++;
        norm += frhs[j][i].u*frhs[j][i].u + frhs[j][i].v*frhs[j][i].v;
      }
    }
    ierr = PetscFree(fz);CHKERRQ(ierr);
    ierr = PetscFree(frhs);CHKERRQ(ierr);
    PetscPrintf(MPI_COMM_WORLD,"    ----- Testing Zero Order evaluation -----\n    If ||F_zos(x) - F_rhs(x)||_2/||F_rhs(x)||_2 is O(1.e-8), ADOL-C function evaluation\n      is probably correct.\n");
    PetscPrintf(MPI_COMM_WORLD,"    ||F_zos(x) - F_rhs(x)||_2/||F_rhs(x)||_2 = %.4e\n",sqrt(diff/norm));
  }

  /*
    Calculate Jacobian using ADOL-C
  */
  if (appctx->sparse) {		// TODO. Generate sparsity pattern with jac_pat and then repeat=1
    PetscInt      nnz,*options;
    unsigned int  *rind,*cind;
    PetscScalar   *values;

    nnz = 10*N;
    ierr = PetscMalloc1(2,&options);CHKERRQ(ierr);
    options[0] = 0;options[1] = 0;
    ierr = PetscMalloc1(nnz,&rind);CHKERRQ(ierr);
    ierr = PetscMalloc1(nnz,&cind);CHKERRQ(ierr);
    ierr = PetscMalloc1(nnz,&values);CHKERRQ(ierr);
    sparse_jac(1,N,N,0,u_vec,&nnz,&rind,&cind,&values,options);
    ierr = PetscFree(values);CHKERRQ(ierr);
    ierr = PetscFree(rind);CHKERRQ(ierr);
    ierr = PetscFree(cind);CHKERRQ(ierr);
    ierr = PetscFree(options);CHKERRQ(ierr);

  } else {

    J = myalloc2(N,N);
    jacobian(1,N,N,u_vec,J);
    ierr = PetscFree(u_vec);CHKERRQ(ierr);

    // Insert entries one-by-one
    for(j=0;j<N;j++){
      for(i=0;i<N;i++){
        if(fabs(J[j][i])!=0.){
            col[0] = i; // TODO: better to insert row-by-row, similarly as with the stencil
            ierr = MatSetValues(A,1,&j,1,col,&J[j][i],INSERT_VALUES);CHKERRQ(ierr);
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
