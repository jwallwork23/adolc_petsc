static char help[] = "Demonstrates automatic Jacobian generation using ADOL-C for a time-dependent PDE in 2d, solved using implicit timestepping.\n";

/*
  See ex5.c for details on the equation.

  Here implicit Crank-Nicolson timestepping is used to solve the same problem as in ex5.c. Another key difference is that functions and Jacobians may optionally be calculated in a local sense, with the command line option -local. The the local implementations DMTSComputeIFunction and DMTSComputeIJacobian are passed to the TS solver using DMTSSetIFunctionLocal and DMTSSetIJacobianLocal.
 */

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>	// Include ADOL-C
#include <adolc/adolc_sparse.h>	// Include ADOL-C sparse drivers

#define tag 1

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} AField;

typedef struct {
  PetscReal   D1,D2,gamma,kappa;
  PetscBool   no_an,sparse,sparse_view;
  AField      **u_a,**f_a;
  PetscScalar **Seed,**Rec;
  PetscInt    p;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*),InitialConditions(DM,Vec);
extern PetscErrorCode IJacobianByHand(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
extern PetscErrorCode IJacobianADOLC(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
extern PetscErrorCode DMTSComputeIFunction(DM,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode DMTSComputeIJacobianByHand(DM,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
extern PetscErrorCode DMTSComputeIJacobianADOLC(DM,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
extern PetscErrorCode ILocalPassive(DM da,Field **f,Field **u,Field **udot,void *ptr);
extern PetscErrorCode ILocalActive(DM da,Field **f,Field **u,Field **udot,void *ptr);

/* Utility functions for automatic Jacobian computation */
extern PetscErrorCode AFieldCreate2d(DM da,AField *cgs,AField **a2d);
extern PetscErrorCode AFieldGiveGhostPoints2d(DM da,AField *cgs,AField **a2d[]);
extern PetscErrorCode AFieldDestroy2d(DM da,AField *cgs[],AField **a2d[]);
extern PetscErrorCode PrintMat(MPI_Comm comm,const char* name,PetscInt m,PetscInt n,PetscScalar **M);
extern PetscErrorCode PrintSparsity(MPI_Comm comm,PetscInt m,unsigned int **JP);
extern PetscErrorCode GetColoring(DM da,PetscInt m,PetscInt n,unsigned int **JP,ISColoring *iscoloring);
extern PetscErrorCode CountColors(ISColoring iscoloring,PetscInt *p);
extern PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring,PetscScalar **Seed);
extern PetscErrorCode GetRecoveryMatrix(PetscScalar **Seed,unsigned int **JP,PetscInt m,PetscInt p,PetscScalar **Rec);
extern PetscErrorCode RecoverJacobian(Mat J,PetscInt m,PetscInt p,PetscScalar **Rec,PetscScalar **Jcomp);

int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x,r,xdot;            /* solution, residual, time derivative */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  PetscBool      local = PETSC_FALSE;
  PetscInt       xs,ys,xm,ym,gxs,gys,gxm,gym,i,m,n,p,dofs = 2,ctrl[3] = {0,0,0};
  AField         **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;
  PetscScalar    **Seed = NULL,**Rec = NULL,*u_vec;
  unsigned int   **JP = NULL;
  ISColoring     iscoloring;
  PetscBool      byhand = PETSC_FALSE;
  MPI_Comm       comm = MPI_COMM_WORLD;


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  appctx.no_an = PETSC_FALSE;appctx.sparse = PETSC_FALSE;appctx.sparse_view = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse",&appctx.sparse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse_view",&appctx.sparse_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&appctx.no_an,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-local",&local,NULL);CHKERRQ(ierr);
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
  ierr = VecDuplicate(x,&xdot);CHKERRQ(ierr); // FIXME: What to put?

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
  ierr = TSSetApplicationContext(ts,&appctx);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  if (!local) {
    ierr = TSSetIFunction(ts,NULL,IFunction,&appctx);CHKERRQ(ierr);
  }
  else {
    ierr = DMTSSetIFunctionLocal(da,DMTSComputeIFunction,&appctx);CHKERRQ(ierr);
  }

  /*
    In the case where ADOL-C generates the Jacobian in compressed format, seed and recovery matrices
    are required. Since the sparsity structure of the Jacobian does not change over the course of the
    time integration, we can save computational effort by only generating these objects once.
  */
  if ((appctx.sparse) && (!appctx.no_an)) {

    ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    m = dofs*gxm*gym;  // Number of dependent variables
    n = m;             // Number of independent variables

    // Trace RHSFunction, so that ADOL-C has tape to read from
    ierr = PetscMalloc1(n,&u_vec);CHKERRQ(ierr);
    ierr = IFunction(ts,1.0,x,r,xdot,&appctx);CHKERRQ(ierr); // Need use IFunction and give xdot
    exit(1); // FIXME: What to put for xdot?

    // Generate sparsity pattern and create an associated colouring
    JP = (unsigned int **) malloc(m*sizeof(unsigned int*));
    jac_pat(tag,m,n,u_vec,JP,ctrl);
    ierr = GetColoring(da,m,n,JP,&iscoloring);CHKERRQ(ierr);
    ierr = CountColors(iscoloring,&p);CHKERRQ(ierr);

    // Generate seed matrix
    Seed = myalloc2(n,p);
    ierr = GenerateSeedMatrix(iscoloring,Seed);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

    // Generate recovery matrix
    Rec = myalloc2(m,p);
    ierr = GetRecoveryMatrix(Seed,JP,m,p,Rec);CHKERRQ(ierr);
    if (appctx.sparse_view) {
      ierr = PrintSparsity(comm,m,JP);CHKERRQ(ierr);
      ierr = PrintMat(comm,"Seed matrix:",n,p,Seed);CHKERRQ(ierr);
    }

    // Store results and free workspace
    appctx.Seed = Seed;
    appctx.Rec = Rec;
    appctx.p = p;
    for (i=0;i<m;i++)
      free(JP[i]);
    free(JP);
    ierr = PetscFree(u_vec);CHKERRQ(ierr);
  }
  if (!local) {
    if (!byhand) {
      ierr = TSSetIJacobian(ts,NULL,NULL,IJacobianADOLC,&appctx);CHKERRQ(ierr);
    } else {
      ierr = TSSetIJacobian(ts,NULL,NULL,IJacobianByHand,&appctx);CHKERRQ(ierr);
    }
  }
  else {
    if (!byhand) {
      ierr = DMTSSetIJacobianLocal(da,DMTSComputeIJacobianADOLC,&appctx);CHKERRQ(ierr);
    } else {
      ierr = DMTSSetIJacobianLocal(da,DMTSComputeIJacobianByHand,&appctx);CHKERRQ(ierr);
    }
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
  ierr = TSSetTimeStep(ts,0.5);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space and call destructors for AFields.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if (appctx.sparse) {
    myfree2(Rec);
    myfree2(Seed);
  }
  if (!appctx.no_an) {
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
  PetscInt       xm,ym;
  Field          **u,**f,**udot;
  Vec            localU,localF;

  // FIXME: localUdot???

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
  ierr = DMDAVecGetArrayRead(da,Udot,&udot);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,NULL,NULL,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  if (!appctx->no_an) {
    ierr = ILocalActive(da,f,u,udot,appctx);CHKERRQ(ierr);
  } else {
    ierr = ILocalPassive(da,f,u,udot,appctx);CHKERRQ(ierr);
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
  ierr = DMDAVecRestoreArrayRead(da,Udot,&udot);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localF);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ILocalActive(DM da,Field **f,Field **u,Field **udot,void *ptr)
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

    TODO: Should there be independence of udot?
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_a[j][i].u <<= u[j][i].u;
      u_a[j][i].v <<= u[j][i].v;
    }
  }

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u_a[j][i].u;
      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;
      vc        = u_a[j][i].v;
      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
      f_a[j][i].u = udot[j][i].u - ( appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc) );
      f_a[j][i].v = udot[j][i].v - ( appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc );
    }
  }
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);

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

PetscErrorCode ILocalPassive(DM da,Field **f,Field **u,Field **udot,void *ptr)
{
  PetscErrorCode  ierr;
  AppCtx          *appctx = (AppCtx*)ptr;
  PetscInt        i,j,xs,ys,xm,ym,gxs,gys,gxm,gym,Mx,My;
  PetscReal       hx,hy,sx,sy;
  PetscScalar     uc,uxx,uyy,vc,vxx,vyy;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
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
      f[j][i].u = udot[j][i].u - ( appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc) );
      f[j][i].v = udot[j][i].v - ( appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc );
    }
  }
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);
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
      stencil[0].c = 1; entries[0] = -appctx->D2*sy;
      stencil[1].c = 1; entries[1] = -appctx->D2*sy;
      stencil[2].c = 1; entries[2] = -appctx->D2*sx;
      stencil[3].c = 1; entries[3] = -appctx->D2*sx;
      stencil[4].c = 1; entries[4] = 2.0*appctx->D2*(sx + sy) - 2.0*uc*vc + appctx->gamma + appctx->kappa + a;
      stencil[5].c = 0; entries[5] = -vc*vc;
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


PetscErrorCode IJacobianADOLC(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*)
{
  PetscFunctionBegin;

  // TODO

  PetscFunctionReturn(0);
}

PetscErrorCode DMTSComputeIFunction(DM da,PetscReal ftime,Vec localU,Vec Udot,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy;
  Field          **u,**f,**udot;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 2.50/(PetscReal)Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)My; sy = 1.0/(hy*hy);

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

  /*
     Compute function over the locally owned part of the grid
  */
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
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Udot,&udot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSComputeIJacobianByHand(DM da,PetscReal t,Vec localU,Vec Udot,PetscReal a,Mat A,Mat BB,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,vc;
  Field          **u;
  MatStencil     stencil[6],rowstencil;
  PetscScalar    entries[6];

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 2.50/(PetscReal)Mx; sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)My; sy = 1.0/(hy*hy);

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
      stencil[0].c = 1; entries[0] = -appctx->D2*sy;
      stencil[1].c = 1; entries[1] = -appctx->D2*sy;
      stencil[2].c = 1; entries[2] = -appctx->D2*sx;
      stencil[3].c = 1; entries[3] = -appctx->D2*sx;
      stencil[4].c = 1; entries[4] = 2.0*appctx->D2*(sx + sy) - 2.0*uc*vc + appctx->gamma + appctx->kappa + a;
      stencil[5].c = 0; entries[5] = -vc*vc;
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
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMTSComputeIJacobianADOLC(DM,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*)
{
  PetscFunctionBegin;

  // TODO

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

PetscErrorCode GetColoring(DM da,PetscInt m,PetscInt n,unsigned int **JP,ISColoring *iscoloring)
{
  PetscErrorCode         ierr;
  Mat                    S;
  MatColoring            coloring;
  PetscInt               i,j,nnz[m],onz[m];
  //ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;

  /*
    Extract number of nonzeros and colours required from JP.
  */
  for (i=0; i<m; i++) {
    nnz[i] = (PetscInt) JP[i][0];
    onz[i] = nnz[i];
    for (j=1; j<=nnz[i]; j++) {
      if (i == (PetscInt) JP[i][j])
        onz[i]--;
    }
  }

  /*
     Preallocate nonzeros by specifying local-to-global mapping. 

     NOTE: Using DMCreateMatrix overestimates nonzeros.
  */
  //ierr = DMCreateMatrix(da,&S);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_SELF,m,n,PETSC_DETERMINE,PETSC_DETERMINE,0,nnz,0,onz,&S);CHKERRQ(ierr);
  ierr = MatSetFromOptions(S);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);             // FIXME: Colouring doesn't seem right
  //ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  //ierr = MatSetLocalToGlobalMapping(S,NULL,NULL);
  ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  //ierr = MatView(S,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
    Extract colouring, with smallest last ('sl') as default.

    NOTE: Use -mat_coloring_type <sl,lf,id,natural,greedy,jp> to change mode.
    FIXME: Only natural is currently working.
  */
  ierr = MatColoringCreate(S,&coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring,iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode CountColors(ISColoring iscoloring,PetscInt *p) 
{
  PetscErrorCode ierr;
  IS             *is;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,p,&is);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring,PetscScalar **Seed)
{
  PetscErrorCode ierr;
  IS             *is;
  PetscInt       p,size,i,j;
  const PetscInt *indices;

  PetscFunctionBegin;

  ierr = ISColoringGetIS(iscoloring,&p,&is);CHKERRQ(ierr);
  for (i=0; i<p; i++) {
    ierr = ISGetLocalSize(is[i],&size);CHKERRQ(ierr);
    ierr = ISGetIndices(is[i],&indices);CHKERRQ(ierr);
    for (j=0; j<size; j++)
      Seed[indices[j]][i] = 1.;
    ierr = ISRestoreIndices(is[i],&indices);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GetRecoveryMatrix(PetscScalar **Seed,unsigned int **JP,PetscInt m,PetscInt p,PetscScalar **Rec)
{
  PetscInt i,j,k,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0; colour<p; colour++) {
      Rec[i][colour] = -1.;
      for (k=1; k<=(PetscInt) JP[i][0]; k++) {
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
    for (colour=0; colour<p; colour++) {
      j = (PetscInt) Rec[i][colour];
      if (j != -1)
        ierr = MatSetValuesLocal(J,1,&i,1,&j,&Jcomp[i][colour],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
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
      output_file: output/ex5imp_1.out
      timeoutfactor: 3

   test:
      suffix: 3
      args: -ts_view -ts_monitor -ts_max_time 500 -local
      requires: double
      timeoutfactor: 3

TEST*/
