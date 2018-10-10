static char help[] = "Demonstrates automatic, matrix-free Jacobian generation using ADOL-C for a time-dependent PDE in 2d, solved using implicit timestepping.\n";

/*
  See ex5.c for details on the equation.

  Here implicit Crank-Nicolson timestepping is used to solve the same problem as in ex5.c. Another
  key difference is that functions are calculated in a local sense, using the local implementations
  IFunctionLocalPassive and IFunctionLocalActive, as in ex5imp. These are passed to the TS solver
  using DMTSSetIFunctionLocal. The Jacobian is generated matrix-free using MyMult, which overloads the
  MatMult operation. The function IJacobianMatFree acts to pass TS context information to the
  matrix-free context.

  Credit for the non-AD implementation to Hong Zhang.
*/

#include "utils/jacobian.cpp"

#define tag 1

/*
   User-defined routines
*/
static PetscErrorCode MyMult(Mat,Vec,Vec);
static PetscErrorCode JacobianVectorProduct(Mat,Vec,Vec);

int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x;                   /* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;              /* Application context */
  MatAppCtx      matctx;              /* Matrix (free) context */
  Mat            A;                   /* Jacobian matrix */
  PetscInt       gys,gxm,gym;
  AField         **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;
  PetscBool      byhand = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char*)0,help);
  PetscFunctionBeginUser;
  appctx.no_an = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-jacobian_by_hand",&byhand,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotation",&appctx.no_an,NULL);CHKERRQ(ierr);
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create matrix free context
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetMatType(da,MATSHELL);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatShellSetContext(A,&matctx);CHKERRQ(ierr);
  if (byhand) {
    ierr = MatShellSetOperation(A,MATOP_MULT,(void (*)(void))MyMult);CHKERRQ(ierr);
    ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
    ierr = DMCreateMatrix(da,&appctx.Jac);CHKERRQ(ierr);
  } else {
    ierr = MatShellSetOperation(A,MATOP_MULT,(void (*)(void))JacobianVectorProduct);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(x,&matctx.X);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&matctx.Xdot);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  if (appctx.no_an) {
    ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)IFunctionLocalPassive,&appctx);CHKERRQ(ierr);
  } else {
    ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)IFunctionLocalActive,&appctx);CHKERRQ(ierr);
  }

  if (!appctx.no_an) {

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
    appctx.m = 2*gxm*gym;
    appctx.n = 2*gxm*gym;

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
     Set Jacobian. In this case, IJacobianMatFree simply acts to pass context
     information to the matrix-free Jacobian vector product.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetIJacobian(ts,A,A,IJacobianMatFree,&appctx);CHKERRQ(ierr);

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
  ierr = VecDestroy(&matctx.X);CHKERRQ(ierr);
  ierr = VecDestroy(&matctx.Xdot);CHKERRQ(ierr);
  if (byhand) {
    ierr = MatDestroy(&appctx.Jac);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

static PetscErrorCode MyMult(Mat A_shell,Vec X,Vec Y)
{
  MatAppCtx         *mctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A_shell,(void**)&mctx);CHKERRQ(ierr);
  ierr = RHSJacobianByHand(mctx->ts,mctx->time,mctx->X,mctx->actx->Jac,mctx->actx->Jac,mctx->actx);CHKERRQ(ierr);
  ierr = MatScale(mctx->actx->Jac,-1);CHKERRQ(ierr);
  ierr = MatShift(mctx->actx->Jac,mctx->shift);CHKERRQ(ierr);
  ierr = MatMult(mctx->actx->Jac,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  For an implicit Jacobian we may use the rule that
     G = M*xdot - f(x)    ==>     dG/dx = a*M - df/dx,
  where a = d(xdot)/dx is a constant. Evaluated at x0 and acting upon a vector x1:
     (dG/dx)(x0) * x1 = (a*M - df/dx)(x0) * x1.
*/
static PetscErrorCode JacobianVectorProduct(Mat A_shell,Vec X,Vec Y)
{
  MatAppCtx            *mctx;
  PetscErrorCode    ierr;
  PetscInt          m,n,i;
  const PetscScalar *x0;
  PetscScalar       *action,*x1;
  Vec               localX0,localX1;
  DM                da;

  PetscFunctionBegin;

  /* Get matrix-free context info */
  ierr = MatShellGetContext(A_shell,(void**)&mctx);CHKERRQ(ierr);
  m = mctx->actx->m;
  n = mctx->actx->n;

  /* Get local input vectors and extract data, x0 and x1*/
  ierr = TSGetDM(mctx->ts,&da);CHKERRQ(ierr);

  ierr = DMGetLocalVector(da,&localX0);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,mctx->X,INSERT_VALUES,localX0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,mctx->X,INSERT_VALUES,localX0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX1);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX1);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,localX0,&x0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localX1,&x1);CHKERRQ(ierr);

  /* First, calculate action of the -df/dx part using ADOL-C */
  ierr = PetscMalloc1(m,&action);CHKERRQ(ierr);
  fos_forward(tag,m,n,0,x0,x1,NULL,action);	// TODO: Could replace NULL to implement ZOS test

  // TODO: temp --------------------------------------
  PetscInt xs,ys,xm,ym,gxs,gys,gxm,gym,d,j,k = 0;
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);

  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      for (d=0; d<2; d++) {
        if ((i >= xs) && (i < xs+xm) && (j >= ys) && (j < ys+ym)) { 
          ierr = VecSetValuesLocal(Y,1,&k,&action[k],INSERT_VALUES);CHKERRQ(ierr);
        }
        k++;
      }
    }
  }
  // -------------------------------------------------- 
/*
  for (i=0; i<m; i++) {
    ierr = VecSetValuesLocal(Y,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }
*/
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(Y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(Y);CHKERRQ(ierr);

  /* Second, shift by action of a*M */ 
  ierr = VecAXPY(Y,mctx->shift,X);CHKERRQ(ierr);

  /* Restore local vector */
  ierr = DMDAVecRestoreArray(da,localX1,&x1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localX0,&x0);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX1);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

