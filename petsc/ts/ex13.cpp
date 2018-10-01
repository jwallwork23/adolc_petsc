

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
#include <adolc/adolc_sparse.h> // Include ADOL-C sparse drivers

#define tag 1

/*
   User-defined data structures
*/
typedef struct {
  PetscReal   c;
  PetscBool   zos,zos_view,no_an,sparse,sparse_view;
  adouble     **u_a,**f_a;
  PetscScalar **Seed,**Rec; /* Jacobian seed and recovery matrices */
  PetscInt    p;
} AppCtx;

/* (Slightly modified) functions included in original code of ex13.c */
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSLocalPassive(DM da,PetscScalar **f,PetscScalar **uarray,void *ptr);
extern PetscErrorCode RHSJacobianByHand(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode FormInitialSolution(DM,Vec,void*);

/* Problem specific functions for the purpose of automatic Jacobian computation */
extern PetscErrorCode RHSJacobianADOLC(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSLocalActive(DM da,PetscScalar **f,PetscScalar **uarray,void *ptr);

/* Utility functions for automatic Jacobian computation and printing */
extern PetscErrorCode AdoubleGiveGhostPoints2d(DM da,adouble *cgs,adouble **a2d[]);
extern PetscErrorCode PrintMat(MPI_Comm comm,const char* name,PetscInt m,PetscInt n,PetscScalar **M);
extern PetscErrorCode PrintSparsity(MPI_Comm comm,PetscInt m,unsigned int **JP);
extern PetscErrorCode GetColoring(DM da,PetscInt m,PetscInt n,unsigned int **JP,ISColoring *iscoloring);
extern PetscErrorCode CountColors(ISColoring iscoloring,PetscInt *p);
extern PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring,PetscScalar **Seed);
extern PetscErrorCode GetRecoveryMatrix(PetscScalar **Seed,unsigned int **JP,PetscInt m,PetscInt p,PetscScalar **Rec);
extern PetscErrorCode RecoverJacobian(Mat J,PetscInt m,PetscInt p,PetscScalar **Rec,PetscScalar **Jcomp);
extern PetscErrorCode TestZOS2d(DM da,PetscScalar **f,PetscScalar **u,void *ctx);

int main(int argc,char **argv)
{
  TS             ts;                    /* nonlinear solver */
  Vec            u,r;                   /* solution, residual vector */
  Mat            J;                     /* Jacobian matrix */
  PetscInt       steps,xs,ys,xm,ym,gxs,gys,gxm,gym,i,m,n,p,ctrl[3] = {0,0,0};
  PetscErrorCode ierr;
  DM             da;
  PetscReal      ftime,dt;
  AppCtx         user;                  /* user-defined work context */
  adouble        **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;  /* active variables */
  PetscScalar    **Seed = NULL,**Rec = NULL,*u_vec;
  unsigned int   **JP = NULL;
  ISColoring     iscoloring;
  PetscBool      byhand = PETSC_FALSE;
  MPI_Comm       comm = MPI_COMM_WORLD;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  user.no_an = PETSC_FALSE;user.zos = PETSC_FALSE;user.zos_view = PETSC_FALSE;user.sparse = PETSC_FALSE;user.sparse_view = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos",&user.zos,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos_view",&user.zos_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse",&user.sparse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_sparse_view",&user.sparse_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&user.no_an,NULL);CHKERRQ(ierr);

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

/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Allocate memory for (local) active arrays and store references in the
     application context. The active arrays are reused at each active
     section, so need only be created once.

     NOTE: Memory for ADOL-C active variables cannot be allocated using
           PetscMalloc, as this does not call the relevant class
           constructor. Instead, we use the C++ keyword `new`.

           It is also important to deconstruct and free memory appropriately.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (!user.no_an) {

    ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

    // Create contiguous 1-arrays of AFields
    u_c = new adouble[gxm*gym];
    f_c = new adouble[gxm*gym];

    // Corresponding 2-arrays of AFields
    u_a = new adouble*[gym];
    f_a = new adouble*[gym];

    // Align indices between array types and endow ghost points
    ierr = AdoubleGiveGhostPoints2d(da,u_c,&u_a);CHKERRQ(ierr);
    ierr = AdoubleGiveGhostPoints2d(da,f_c,&f_a);CHKERRQ(ierr);

    // Store active variables in context
    user.u_a = u_a;
    user.f_a = f_a;
  }

  if (user.zos) {
    PetscPrintf(comm,"    If ||F_zos(x) - F_rhs(x)||_2/||F_rhs(x)||_2 is O(1.e-8), ADOL-C function evaluation\n      is probably correct.\n");
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,r,RHSFunction,&user);CHKERRQ(ierr);

  /*
    In the case where ADOL-C generates the Jacobian in compressed format, seed and recovery matrices
    are required. Since the sparsity structure of the Jacobian does not change over the course of the
    time integration, we can save computational effort by only generating these objects once.
  */
  if ((user.sparse) && (!user.no_an)) {

    ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
    m = gxm*gym;  // Number of dependent variables
    n = gxm*gym;  // Number of independent variables

    // Trace RHSFunction, so that ADOL-C has tape to read from
    ierr = PetscMalloc1(n,&u_vec);CHKERRQ(ierr);
    ierr = RHSFunction(ts,1.0,u,r,&user);CHKERRQ(ierr);

    // Generate sparsity pattern and create an associated colouring
    JP = (unsigned int **) malloc(m*sizeof(unsigned int*));
    jac_pat(tag,m,n,u_vec,JP,ctrl);
    if (user.sparse_view) {
      ierr = PrintSparsity(comm,m,JP);CHKERRQ(ierr);
    }

    // Extract colouring
    ierr = GetColoring(da,m,n,JP,&iscoloring);CHKERRQ(ierr);
    ierr = CountColors(iscoloring,&p);CHKERRQ(ierr);

    // Generate seed matrix
    Seed = myalloc2(n,p);
    ierr = GenerateSeedMatrix(iscoloring,Seed);CHKERRQ(ierr);
    ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
    if (user.sparse_view) {
      ierr = PrintMat(comm,"Seed matrix:",n,p,Seed);CHKERRQ(ierr);
    }

    // Generate recovery matrix
    Rec = myalloc2(m,p);
    ierr = GetRecoveryMatrix(Seed,JP,m,p,Rec);CHKERRQ(ierr);

    // Store results and free workspace
    user.Seed = Seed;
    user.Rec = Rec;
    user.p = p;
    for (i=0;i<m;i++)
      free(JP[i]);
    free(JP);
    ierr = PetscFree(u_vec);CHKERRQ(ierr);
  }

  /* Set Jacobian */
  ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  if (!byhand) {
    ierr = TSSetRHSJacobian(ts,J,J,RHSJacobianADOLC,NULL);CHKERRQ(ierr);
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
  if (user.sparse) {
    myfree2(Rec);
    myfree2(Seed);
  }
  if (!user.no_an) {
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

  ierr = PetscFinalize();
  return ierr;
}


PetscErrorCode RHSLocalActive(DM da,PetscScalar **f,PetscScalar **uarray,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym,Mx,My;
  PetscReal      hx,hy,sx,sy,two = 2.0;
  adouble        **f_a = user->f_a,**u_a = user->u_a;
  adouble        u,uxx,uyy;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
  hy = 1.0/(PetscReal)(My-1); sy = 1.0/(hy*hy);

  trace_on(tag);  // ----------------------------------------------- Start of active section

  /*
    Mark independence

    NOTE: Ghost points are marked as independent, in place of the points they represent on
          other processors / on other boundaries.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++)
      u_a[j][i] <<= uarray[j][i];
  }

  /*
    Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1)  // Consider boundary cases
        f_a[j][i] = u_a[j][i];
      else {
        u         = u_a[j][i];
        uxx       = (-two*u + u_a[j][i-1] + u_a[j][i+1])*sx;
        uyy       = (-two*u + u_a[j-1][i] + u_a[j+1][i])*sy;
        f_a[j][i] = uxx + uyy;
      }
    }
  }

  /*
    Mark dependence

    NOTE: Ghost points are marked as dependent in order to vastly simplify index notation
          during Jacobian assembly.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++)
      f_a[j][i] >>= f[j][i];
  }
  trace_off();  // ----------------------------------------------- End of active section

  PetscFunctionReturn(0);
}

PetscErrorCode RHSLocalPassive(DM da,PetscScalar **f,PetscScalar **uarray,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscReal      hx,hy,sx,sy,two=2.0;
  PetscScalar    u,uxx,uyy;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 1.0/(PetscReal)(Mx-1); sx = 1.0/(hx*hx);
  hy = 1.0/(PetscReal)(My-1); sy = 1.0/(hy*hy);
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        f[j][i] = uarray[j][i];
        continue;
      }
      u       = uarray[j][i];
      uxx     = (-two*u + uarray[j][i-1] + uarray[j][i+1])*sx;
      uyy     = (-two*u + uarray[j-1][i] + uarray[j+1][i])*sy;
      f[j][i] = uxx + uyy;
    }
  }

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   RHSFunction - Evaluates nonlinear function, F(u).

   Input Parameters:
.  ts - the TS context
.  U - input vector
.  ptr - optional user-defined context, as set by TSSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode RHSFunction(TS ts,PetscReal ftime,Vec U,Vec F,void *ptr)
{
  AppCtx         *user=(AppCtx*)ptr;
  DM             da;
  PetscErrorCode ierr;
  PetscInt       xm,ym;
  PetscScalar    **u,**f;
  Vec            localU,localF;

  PetscFunctionBeginUser;
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
  ierr = DMGlobalToLocalBegin(da,F,INSERT_VALUES,localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,F,INSERT_VALUES,localF);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localF,&f);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,NULL,NULL,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  if (!user->no_an) {
    ierr = RHSLocalActive(da,f,u,user);CHKERRQ(ierr);

    /* Test zeroth order scalar evaluation in ADOL-C gives the same result */
    if (user->zos) {
      ierr = TestZOS2d(da,f,u,user);CHKERRQ(ierr);
    }
  } else {
    ierr = RHSLocalPassive(da,f,u,user);CHKERRQ(ierr);
  }

  /*
     Gather global vector, using the 2-step process
        DMLocalToGlobalBegin(),DMLocalToGlobalEnd().
  */
  ierr = DMLocalToGlobalBegin(da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,localF,INSERT_VALUES,F);CHKERRQ(ierr);

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,localF,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localF);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = PetscLogFlops(11.0*xm*ym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
   RHSJacobianByHand - User-provided routine to compute the Jacobian of
   the nonlinear right-hand-side function of the ODE, as given in ex13.c.

   Input Parameters:
   ts - the TS context
   t - current time
   U - global input vector
   ctx - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   J - Jacobian matrix
   Jpre - optionally different preconditioning matrix
*/
PetscErrorCode RHSJacobianByHand(TS ts,PetscReal t,Vec U,Mat J,Mat Jpre,void *ctx)
{
  PetscErrorCode ierr;
  DM             da;
  DMDALocalInfo  info;
  PetscInt       i,j;
  PetscReal      hx,hy,sx,sy;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  hx   = 1.0/(PetscReal)(info.mx-1); sx = 1.0/(hx*hx);
  hy   = 1.0/(PetscReal)(info.my-1); sy = 1.0/(hy*hy);
  for (j=info.ys; j<info.ys+info.ym; j++) {
    for (i=info.xs; i<info.xs+info.xm; i++) {
      PetscInt    nc = 0;
      MatStencil  row,col[5];
      PetscScalar val[5];
      row.i = i; row.j = j;
      if (i == 0 || j == 0 || i == info.mx-1 || j == info.my-1) {
        col[nc].i = i; col[nc].j = j; val[nc++] = 1.0;
      } else {
        col[nc].i = i-1; col[nc].j = j;   val[nc++] = sx;
        col[nc].i = i+1; col[nc].j = j;   val[nc++] = sx;
        col[nc].i = i;   col[nc].j = j-1; val[nc++] = sy;
        col[nc].i = i;   col[nc].j = j+1; val[nc++] = sy;
        col[nc].i = i;   col[nc].j = j;   val[nc++] = -2*sx - 2*sy;
      }
      ierr = MatSetValuesStencil(Jpre,1,&row,nc,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------- */
/*
   RHSJacobianADOLC - Automatically generated ADOL-C version.

   Input Parameters:
   ts - the TS context
   t - current time
   U - global input vector
   ctx - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   J - Jacobian matrix
   Jpre - optionally different preconditioning matrix
*/
PetscErrorCode RHSJacobianADOLC(TS ts,PetscReal t,Vec U,Mat J,Mat Jpre,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  PetscErrorCode ierr;
  DM             da;
  PetscInt       i,j,k = 0,gxs,gys,gxm,gym,m,n;
  PetscScalar    **u,*u_vec,**Jac = NULL,*f_vec;
  Vec            localU;
  MPI_Comm       comm = MPI_COMM_WORLD;

  PetscFunctionBeginUser;
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

  /* Get ghosted grid boundaries */
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  m = gxm*gym;  // Number of dependent variables
  n = m;        // Number of independent variables

  /* Convert 2-array to a 1-array, so this can be read by ADOL-C */
  ierr = PetscMalloc1(n,&u_vec);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++)
      u_vec[k++] = u[j][i];
  }

  /*
    Calculate Jacobian using ADOL-C
  */

  if (appctx->sparse) {

    /*
      Compute Jacobian in compressed format and recover from this, using seed and recovery matrices
      computed earlier.
    */
    ierr = PetscMalloc1(m,&f_vec);CHKERRQ(ierr);
    Jac = myalloc2(m,appctx->p);
    fov_forward(tag,m,n,appctx->p,u_vec,appctx->Seed,f_vec,Jac);
    ierr = PetscFree(f_vec);CHKERRQ(ierr);
    if (appctx->sparse_view) {
      ierr = TSGetStepNumber(ts,&k);
      if (k == 0) {
        ierr = PrintMat(comm,"Compressed Jacobian:",m,appctx->p,Jac);CHKERRQ(ierr);
      }
    }
    ierr = RecoverJacobian(J,m,appctx->p,appctx->Rec,Jac);CHKERRQ(ierr);
    myfree2(Jac);

  } else {

    /*
      Default method of computing full Jacobian (not recommended!).
    */
    Jac = myalloc2(m,n);
    jacobian(tag,m,n,u_vec,Jac);
    ierr = PetscFree(u_vec);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(Jac[i][j]) > 1.e-16)	// TODO: Instead use where nonzeros expected
          ierr = MatSetValuesLocal(J,1,&i,1,&j,&Jac[i][j],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    myfree2(Jac);
  }

  /*
    Restore vectors
  */
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);

  /*
    Assemble local matrix
  */
  ierr = MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da,Vec U,void* ptr)
{
  AppCtx         *user=(AppCtx*)ptr;
  PetscReal      c=user->c;
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym,Mx,My;
  PetscScalar    **u;
  PetscReal      hx,hy,x,y,r;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  hx = 1.0/(PetscReal)(Mx-1);
  hy = 1.0/(PetscReal)(My-1);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);

  /* Get local grid boundaries */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    y = j*hy;
    for (i=xs; i<xs+xm; i++) {
      x = i*hx;
      r = PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5));
      if (r < .125) u[j][i] = PetscExpReal(c*r*r*r);
      else u[j][i] = 0.0;
    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Shift indices in adouble array to endow it with ghost points.
*/
PetscErrorCode AdoubleGiveGhostPoints2d(DM da,adouble *cgs,adouble **a2d[])
{
  PetscErrorCode ierr;
  PetscInt       gxs,gys,gxm,gym,j;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  for (j=0; j<gym; j++)
    (*a2d)[j] = cgs + j*gxm - gxs;
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
  PetscInt               i,j,k,nnz[m],onz[m];
  PetscScalar            one = 1.;

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
     Preallocate nonzeros as ones. 

     NOTE: Using DMCreateMatrix overestimates nonzeros.
  */
  ierr = MatCreateAIJ(PETSC_COMM_SELF,m,n,PETSC_DETERMINE,PETSC_DETERMINE,0,nnz,0,onz,&S);CHKERRQ(ierr);
  ierr = MatSetFromOptions(S);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);		// FIXME: Colouring doesn't seem right
  for (i=0; i<m; i++) {
    for (j=1; j<=nnz[i]; j++) {
      k = JP[i][j];
      ierr = MatSetValues(S,1,&i,1,&k,&one,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
    Extract colouring, with smallest last ('sl') as default.

    NOTE: Use -mat_coloring_type <sl,lf,id,natural,greedy,jp> to change mode.
    FIXME: jp and greedy not currently working
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
      for (k=1; k<=(PetscInt) JP[i][0];k++) {
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

PetscErrorCode TestZOS2d(DM da,PetscScalar **f,PetscScalar **u,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       m,n,gxs,gys,gxm,gym,i,j,k = 0;
  PetscScalar    diff = 0,norm = 0,*u_vec,*fz;
  MPI_Comm       comm = MPI_COMM_WORLD;

  PetscFunctionBegin;

  /* Get extent of region owned by processor */
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  m = gxm*gym;
  n = m;

  /* Convert to a 1-array */
  ierr = PetscMalloc1(n,&u_vec);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++)
      u_vec[k++] = u[j][i];
  }
  k = 0;

  /* Zero order scalar evaluation vs. calling RHS function */
  ierr = PetscMalloc1(m,&fz);CHKERRQ(ierr);
  zos_forward(tag,m,n,0,u_vec,fz);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      if ((appctx->zos_view) && ((fabs(f[j][i]) > 1.e-16) || (fabs(fz[k]) > 1.e-16)))
        PetscPrintf(comm,"(%2d,%2d): F_rhs = %+.4e, F_zos = %+.4e\n",j,i,f[j][i],fz[k]);
      diff += (f[j][i]-fz[k])*(f[j][i]-fz[k]);k++;
      norm += f[j][i]*f[j][i];
    }
  }
  diff = sqrt(diff);
  norm = diff/sqrt(norm);
  PetscPrintf(comm,"    ----- Testing Zero Order evaluation -----\n");
  PetscPrintf(comm,"    ||Fzos - Frhs||_2/||Frhs||_2 = %.4e, ||Fzos - Frhs||_2 = %.4e\n",norm,diff);
  ierr = PetscFree(fz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
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

