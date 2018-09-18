static char help[] = "Demonstrates adjoint sensitivity analysis for Reaction-Diffusion Equations.\n";

/*
      See ex5.c for details on the equations.
      This code applies the operator overloading automatic differentiation techniques provided by ADOL-C to automatically generate Jacobians for nonlinear partial differential equations (PDEs) and associated adjoint equations. Whilst doing this is unnecessary for equations such as these, where the Jacobian can be derived quite easily, automatic Jacobian generation would be greatly beneficial for more complex PDEs.
      Handcoded Jacobians are included here for comparison.

  Runtime options:
    -forwardonly      - run the forward simulation without adjoint
    -implicitform     - provide IFunction and IJacobian to TS, if not set, RHSFunction and RHSJacobian will be used
    -aijpc            - set the preconditioner matrix to be aij (the Jacobian matrix can be of a different type such as ELL)
    -jacobian_by_hand - use a hand-coded Jacobian
    -no_annotation    - do not annotate using ADOL-C
    -sparse           - generate Jacobian using ADOL-C sparse Jacobian driver TODO
 */

#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} AField;

typedef struct {
  PetscReal D1,D2,gamma,kappa;
  PetscBool aijpc,no_an;
  PetscInt  Mx,My;
  AField    **u_a,**f_a;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*),InitialConditions(DM,Vec);
extern PetscErrorCode RHSJacobianADOLC(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSJacobianByHand(TS,PetscReal,Vec,Mat,Mat,void*);

extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode IJacobianADOLC(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
extern PetscErrorCode IJacobianByHand(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);

PetscErrorCode InitializeLambda(DM da,Vec lambda,PetscReal x,PetscReal y)
{
   PetscInt i,j,Mx,My,xs,ys,xm,ym;
   PetscErrorCode ierr;
   Field **l;
   PetscFunctionBegin;

   ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
   /* locate the global i index for x and j index for y */
   i = (PetscInt)(x*(Mx-1));
   j = (PetscInt)(y*(My-1));
   ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

   if (xs <= i && i < xs+xm && ys <= j && j < ys+ym) {
     /* the i,j vertex is on this process */
     ierr = DMDAVecGetArray(da,lambda,&l);CHKERRQ(ierr);
     l[j][i].u = 1.0;
     l[j][i].v = 1.0;
     ierr = DMDAVecRestoreArray(da,lambda,&l);CHKERRQ(ierr);
   }
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

int main(int argc,char **argv)
{
  TS             ts;                  		/* ODE integrator */
  Vec            x;                   		/* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  AField         **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;
  PetscInt       gxs,gys,gxm,gym;
  Vec            lambda[1];
  PetscScalar    *x_ptr;
  PetscBool      forwardonly=PETSC_FALSE,implicitform=PETSC_FALSE,byhand=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-forwardonly",&forwardonly,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-implicitform",&implicitform,NULL);CHKERRQ(ierr);
  appctx.aijpc = PETSC_FALSE,appctx.no_an = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-aijpc",&appctx.aijpc,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&appctx.no_an,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,64,64,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,NULL,&da);CHKERRQ(ierr);
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  if (!implicitform) {
    ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);
    if (!byhand) {
      ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobianADOLC,&appctx);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobianByHand,&appctx);CHKERRQ(ierr);
    }
  } else {
    ierr = TSSetIFunction(ts,NULL,IFunction,&appctx);CHKERRQ(ierr);
    if (appctx.aijpc) {
      Mat                    A,B;

      ierr = DMSetMatType(da,MATSELL);CHKERRQ(ierr);
      ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
      ierr = MatConvert(A,MATAIJ,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
      /* FIXME do we need to change viewer to display matrix in natural ordering as DMCreateMatrix_DA does? */
      if (!byhand) {
        ierr = TSSetIJacobian(ts,A,B,IJacobianADOLC,&appctx);CHKERRQ(ierr);
      } else {
        ierr = TSSetIJacobian(ts,A,B,IJacobianByHand,&appctx);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&A);CHKERRQ(ierr);
      ierr = MatDestroy(&B);CHKERRQ(ierr);
    } else {
      if (!byhand) {
        ierr = TSSetIJacobian(ts,NULL,NULL,IJacobianADOLC,&appctx);CHKERRQ(ierr);
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

  /*
    Have the TS save its trajectory so that TSAdjointSolve() may be used
  */
  if (!forwardonly) { ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr); }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,200.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.5);CHKERRQ(ierr);
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
    ierr = VecGetArray(lambda[0],&x_ptr);CHKERRQ(ierr);
    ierr = InitializeLambda(da,lambda[0],0.5,0.5);CHKERRQ(ierr);
    ierr = TSSetCostGradients(ts,1,lambda,NULL);CHKERRQ(ierr);
    ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
    ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  }
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

/* ------------------------------------------------------------------- */
/*
   RHSFunction - Evaluates nonlinear function, F(x).

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
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal      hx,hy,sx,sy;
  Field          **u,**f;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)appctx->Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)appctx->My; sy = 1.0/(hy*hy);

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
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  if (!appctx->no_an) {
    adouble    uc,uxx,uyy,vc,vxx,vyy;

    trace_on(1);  // ----------------------------------------------- Start of active section

  /*
    Mark independence

    NOTE: Ghost points are marked as independent at this stage, but their contributions to
          the Jacobian will be added to the corresponding rows on other processes, meaning
          the Jacobian remains square.
  */
    for (j=gys; j<gys+gym; j++) {
      for (i=gxs; i<gxs+gxm; i++) {
        appctx->u_a[j][i].u <<= u[j][i].u;appctx->u_a[j][i].v <<= u[j][i].v;
      }
    }

    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {

        // Compute function over the locally owned part of the grid
        uc          = appctx->u_a[j][i].u;
        uxx         = (-2.0*uc + appctx->u_a[j][i-1].u + appctx->u_a[j][i+1].u)*sx;
        uyy         = (-2.0*uc + appctx->u_a[j-1][i].u + appctx->u_a[j+1][i].u)*sy;
        vc          = appctx->u_a[j][i].v;
        vxx         = (-2.0*vc + appctx->u_a[j][i-1].v + appctx->u_a[j][i+1].v)*sx;
        vyy         = (-2.0*vc + appctx->u_a[j-1][i].v + appctx->u_a[j+1][i].v)*sy;
        appctx->f_a[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);
        appctx->f_a[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc;

        // Mark dependence
        appctx->f_a[j][i].u >>= f[j][i].u;appctx->f_a[j][i].v >>= f[j][i].v;
      }
    }
    trace_off();  // ----------------------------------------------- End of active section
  } else {
    PetscScalar    uc,uxx,uyy,vc,vxx,vyy;

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

  hx = 2.5/(PetscReal)Mx;
  hy = 2.5/(PetscReal)My;

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
  PetscInt       i,j,k = 0,w_ghost = 0,xs,ys,xm,ym,gxs,gys,gxm,gym,L,G;
  PetscInt       loc,d,dofs = 2;
  PetscScalar    *u_vec,**J;
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

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);

  /*
     Get local grid boundaries and total degrees of freedom on this process
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  /*
    Convert array of structs to a 1-array, so this can be read by ADOL-C
  */
  L = dofs*xm*ym;G = dofs*gxm*gym;
  ierr = PetscMalloc1(G,&u_vec);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_vec[k++] = u[j][i].u;u_vec[k++] = u[j][i].v;
    }
  }

  /*
    Calculate Jacobian using ADOL-C
  */

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
              loc = d+dofs*(i+j*ym);     // TODO: Test this

          // CASE 2: ghost point above local region
          } else if (j >= ym) {

            // Top boundary
            if ((j >= appctx->My) && (i >= 0) && (i < appctx->Mx))
              loc = d+dofs*i;
            else
              loc = d+dofs*(i+j*ym);              // TODO: Test this

          // CASE 3: ghost point left of local region
          } else if (i < xs) {

            // Left boundary
            if ((i < 0) && (j >= 0) && (j < appctx->My))
              loc = d+dofs*(1+j*ym+appctx->Mx+2*i);
            else
              loc = d+dofs*(i+j*ym);   // TODO: Test this

          // CASE 4: ghost point right of local region
          } else if (i >= xm) {

            // Right boundary
            if ((i >= appctx->Mx) && (j >= 0) && (j < appctx->My))
              loc = d+dofs*(j*ym-2*appctx->Mx+2*i);
            else
              loc = d+dofs*(i+j*ym); // TODO: Test this

          // CASE 5: Interior points of local region
          } else
            loc = d+dofs*(i+j*ym);
          if (fabs(J[k][w_ghost])!=0.)
            ierr = MatSetValues(A,1,&k,1,&loc,&J[k][w_ghost],INSERT_VALUES);CHKERRQ(ierr);
          w_ghost++;
        }
      }
    }
    w_ghost = 0;
  }
  myfree2(J);

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
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,vc;
  Field          **u;
  Vec            localU;
  MatStencil     stencil[6],rowstencil;
  PetscScalar    entries[6];

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)appctx->Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)appctx->My; sy = 1.0/(hy*hy);

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
      if (appctx->aijpc) {
        ierr = MatSetValuesStencil(BB,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      }
      stencil[0].c = 1; entries[0] = appctx->D2*sy;
      stencil[1].c = 1; entries[1] = appctx->D2*sy;
      stencil[2].c = 1; entries[2] = appctx->D2*sx;
      stencil[3].c = 1; entries[3] = appctx->D2*sx;
      stencil[4].c = 1; entries[4] = -2.0*appctx->D2*(sx + sy) + 2.0*uc*vc - appctx->gamma - appctx->kappa;
      stencil[5].c = 0; entries[5] = vc*vc;
      rowstencil.c = 1;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      if (appctx->aijpc) {
        ierr = MatSetValuesStencil(BB,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      }
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
  if (appctx->aijpc) {
    ierr = MatAssemblyBegin(BB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetOption(BB,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*
   IFunction - Evaluates implicit nonlinear function, xdot - F(x).

   Input Parameters:
.  ts - the TS context
.  U - input vector
.  Udot - input vector
.  ptr - optional user-defined context, as set by TSSetRHSFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode IFunction(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal      hx,hy,sx,sy;
  Field          **u,**f,**udot;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)appctx->Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)appctx->My; sy = 1.0/(hy*hy);

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
  ierr = DMDAVecGetArrayRead(da,Udot,&udot);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  if (!appctx->no_an) {
    adouble    uc,uxx,uyy,vc,vxx,vyy;

    trace_on(1);  // ----------------------------------------------- Start of active section

    /*
      Mark independence

      NOTE: Ghost points are marked as independent at this stage, but their contributions to
            the Jacobian will be added to the corresponding rows on other processes, meaning
            the Jacobian remains square.
    */
    for (j=gys; j<gys+gym; j++) {
      for (i=gxs; i<gxs+gxm; i++) {
        appctx->u_a[j][i].u <<= u[j][i].u;appctx->u_a[j][i].v <<= u[j][i].v;
      }
    }

    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {

        // Compute function over the locally owned part of the grid
        uc          = appctx->u_a[j][i].u;
        uxx         = (-2.0*uc + appctx->u_a[j][i-1].u + appctx->u_a[j][i+1].u)*sx;
        uyy         = (-2.0*uc + appctx->u_a[j-1][i].u + appctx->u_a[j+1][i].u)*sy;
        vc          = appctx->u_a[j][i].v;
        vxx         = (-2.0*vc + appctx->u_a[j][i-1].v + appctx->u_a[j][i+1].v)*sx;
        vyy         = (-2.0*vc + appctx->u_a[j-1][i].v + appctx->u_a[j+1][i].v)*sy;
        appctx->f_a[j][i].u = udot[j][i].u - ( appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc) );
        appctx->f_a[j][i].v = udot[j][i].v - ( appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc );

        // Mark dependence
        appctx->f_a[j][i].u >>= f[j][i].u;appctx->f_a[j][i].v >>= f[j][i].v;
      }
    }
    trace_off();  // ----------------------------------------------- End of active section
  } else {
    PetscScalar    uc,uxx,uyy,vc,vxx,vyy;

    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        uc        = u[j][i].u;
        uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
        uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;
        vc        = u[j][i].v;
        vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
        vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
        f[j][i].u = udot[j][i].u - ( appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc) );
        f[j][i].v = udot[j][i].v - ( appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc );
      }
    }
  }
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobianADOLC(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat BB,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       i,j,k = 0,w_ghost = 0,xs,ys,xm,ym,gxs,gys,gxm,gym,L,G;
  PetscInt       loc,d,dofs = 2;
  PetscScalar    *u_vec,**J;
  Field          **u;
  Vec            localU,D;

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

  /*
    For an implicit Jacobian we may use the rule that
       G = M*xdot - f(x)    ==>     dG/dx = a*M - df/dx,
    where a = d(xdot)/dx is a constant.
  */

  /*
    First, calculate the -df/dx part using ADOL-C
  */
  J = myalloc2(L,G);
  jacobian(1,L,G,u_vec,J);
  ierr = PetscFree(u_vec);CHKERRQ(ierr);

  /* Add entries one-by-one. TODO: better to add row-by-row, similarly as with the stencil */
  ierr = MatZeroEntries(J);CHKERRQ(ierr);
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
              loc = d+dofs*(i+xm*(ym+j));     // TODO: Test this

          // CASE 2: ghost point above local region
          } else if (j >= ym) {

            // Top boundary
            if ((j >= appctx->My) && (i >= 0) && (i < appctx->Mx))
              loc = d+dofs*i;
            else
              loc = xs+d*dofs*i;              // TODO: Test this

          // CASE 3: ghost point left of local region
          } else if (i < xs) {

            // Left boundary
            if ((i < 0) && (j >= 0) && (j < appctx->My))
              loc = d+dofs*(1+j*ym+appctx->Mx+2*i);
            else
              loc = d+dofs*(i+j*ym);   // TODO: Test this

          // CASE 4: ghost point right of local region
          } else if (i >= xm) {

            // Right boundary
            if ((i >= appctx->Mx) && (j >= 0) && (j < appctx->My))
              loc = d+dofs*(j*ym-2*appctx->Mx+2*i);
            else
              loc = d+dofs*(i+j*ym); // TODO: Test this

          // CASE 5: Interior points of local region
          } else
            loc = d+dofs*(i+j*ym);
          if (fabs(J[k][w_ghost])!=0.)
            ierr = MatSetValues(A,1,&k,1,&loc,&J[k][w_ghost],ADD_VALUES);CHKERRQ(ierr);
          w_ghost++;
        }
      }
    }
    w_ghost = 0;
  }
  myfree2(J);

  /*
    Next, assemble a*M
  */
  ierr = VecDuplicate(U,&D);CHKERRQ(ierr);
  ierr = VecSet(D,a);CHKERRQ(ierr);

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
  ierr = MatDiagonalSet(A,D,ADD_VALUES);     /* Combine a*M and -df/dx parts */
  ierr = VecDestroy(&D);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode IJacobianByHand(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat BB,void *ctx)
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

  hx = 2.50/(PetscReal)Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)My; sy = 1.0/(hy*hy);

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

      stencil[0].i = i; stencil[0].c = 0; entries[0] = -appctx->D1*sy;
      stencil[1].i = i; stencil[1].c = 0; entries[1] = -appctx->D1*sy;
      stencil[2].i = i-1; stencil[2].c = 0; entries[2] = -appctx->D1*sx;
      stencil[3].i = i+1; stencil[3].c = 0; entries[3] = -appctx->D1*sx;
      stencil[4].i = i; stencil[4].c = 0; entries[4] = 2.0*appctx->D1*(sx + sy) + vc*vc + appctx->gamma + a;
      stencil[5].i = i; stencil[5].c = 1; entries[5] = 2.0*uc*vc;
      rowstencil.i = i; rowstencil.c = 0;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      if (appctx->aijpc) {
        ierr = MatSetValuesStencil(BB,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      }
      stencil[0].c = 1; entries[0] = -appctx->D2*sy;
      stencil[1].c = 1; entries[1] = -appctx->D2*sy;
      stencil[2].c = 1; entries[2] = -appctx->D2*sx;
      stencil[3].c = 1; entries[3] = -appctx->D2*sx;
      stencil[4].c = 1; entries[4] = 2.0*appctx->D2*(sx + sy) - 2.0*uc*vc + appctx->gamma + appctx->kappa + a;
      stencil[5].c = 0; entries[5] = -vc*vc;
      rowstencil.c = 1;

      ierr = MatSetValuesStencil(A,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      if (appctx->aijpc) {
        ierr = MatSetValuesStencil(BB,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
      }
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
  if (appctx->aijpc) {
    ierr = MatAssemblyBegin(BB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetOption(BB,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*TEST

   build:
      requires: !complex !single

   test:
      args: -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -da_grid_x 16 -da_grid_y 16
      output_file: output/ex5adj_1.out

   test:
      suffix: 2
      nsize: 2
      args: -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -ksp_monitor_short -da_grid_x 16 -da_grid_y 16 -ts_trajectory_dirname Test-dir -ts_trajectory_file_template test-%06D.cp

   test:
      suffix: 3
      nsize: 2
      args: -ts_max_steps 10 -ts_dt 10 -ts_adjoint_monitor_draw_sensi

   test:
      suffix: knl
      args: -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -ts_trajectory_type memory -ts_trajectory_solution_only 0 -malloc_hbw -ts_trajectory_use_dram 1
      output_file: output/ex5adj_3.out
      requires: knl

   test:
      suffix: sell
      nsize: 4
      args: -forwardonly -ts_max_steps 10 -ts_monitor -snes_monitor_short -dm_mat_type sell -pc_type none
      output_file: output/ex5adj_sell_1.out

   test:
      suffix: sell2
      nsize: 4
      args: -forwardonly -ts_max_steps 10 -ts_monitor -snes_monitor_short -dm_mat_type sell -pc_type mg -pc_mg_levels 2 -mg_coarse_pc_type sor
      output_file: output/ex5adj_sell_2.out

   test:
      suffix: sell3
      nsize: 4
      args: -forwardonly -ts_max_steps 10 -ts_monitor -snes_monitor_short -dm_mat_type sell -pc_type mg -pc_mg_levels 2 -mg_coarse_pc_type bjacobi -mg_levels_pc_type bjacobi
      output_file: output/ex5adj_sell_3.out

   test:
      suffix: sell4
      nsize: 4
      args: -forwardonly -implicitform -ts_max_steps 10 -ts_monitor -snes_monitor_short -dm_mat_type sell -pc_type mg -pc_mg_levels 2 -mg_coarse_pc_type bjacobi -mg_levels_pc_type bjacobi
      output_file: output/ex5adj_sell_4.out

   test:
      suffix: sell5
      nsize: 4
      args: -forwardonly -ts_max_steps 10 -ts_monitor -snes_monitor_short -dm_mat_type sell -aijpc
      output_file: output/ex5adj_sell_5.out

   test:
      suffix: sell6
      args: -ts_max_steps 10 -ts_monitor -ts_adjoint_monitor -ts_trajectory_type memory -ts_trajectory_solution_only 0 -dm_mat_type sell -pc_type jacobi
      output_file: output/ex5adj_sell_6.out

TEST*/
