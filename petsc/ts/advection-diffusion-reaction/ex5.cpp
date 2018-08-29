
static char help[] = "Demonstrates Pattern Formation with Reaction-Diffusion Equations.\n";

/*
     Page 21, Pattern Formation with Reaction-Diffusion Equations

        u_t = D1 (u_xx + u_yy)  - u*v^2 + gamma(1 -u)
        v_t = D2 (v_xx + v_yy)  + u*v^2 - (gamma + kappa)v

    Unlike in the book this uses periodic boundary conditions instead of Neumann
    (since they are easier for finite differences).
*/

/*
      Helpful runtime monitor options:
           -ts_monitor_draw_solution
           -draw_save -draw_save_movie
           -da_grid_x 12 -da_grid_y 12 -ts_max_steps 1 -snes_test_jacobian
           -da_grid_x 12 -da_grid_y 12 -ts_max_steps 1 -snes_test_jacobian -snes_test_jacobian_view
      Follow any such command with "> <filename>.txt" to save output to file.

      Helpful runtime linear solver options:
           -pc_type mg -pc_mg_galerkin pmat -da_refine 1 -snes_monitor -ksp_monitor -ts_view  (note that these Jacobians are so well-conditioned multigrid may not be the best solver)

      Helpful ADOL-C related options:
           -adolc_test_zos (test Zero Order Scalar evaluation)

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
#include "utils.c"		// For modular arithmetic and coordinate mappings

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} aField;

typedef struct {
  PetscReal D1,D2,gamma,kappa;
  PetscBool zos,no_an;
  aField    u_a,f_a;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*),InitialConditions(DM,Vec);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSJacobianByHand(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSLocalActive(Field **f,Field **u,PetscInt xs,PetscInt xm,PetscInt ys,PetscInt ym,PetscReal hx,PetscReal hy,void *ptr);
extern PetscErrorCode RHSLocalPassive(Field **f,Field **u,PetscInt xs,PetscInt xm,PetscInt ys,PetscInt ym,PetscReal hx,PetscReal hy,void *ptr);

int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x;                   /* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  PetscBool      analytic=PETSC_FALSE,no_annotations=PETSC_FALSE,adolc_test_zos=PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  ierr = PetscOptionsGetBool(NULL,NULL,"-analytic",&analytic,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-no_annotations",&no_annotations,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-adolc_test_zos",&adolc_test_zos,NULL);CHKERRQ(ierr);
  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;
  appctx.zos   = adolc_test_zos;
  appctx.no_an = no_annotations;

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

  // TODO: Create aFields out here for efficiency

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

  ierr = PetscFinalize();
  return ierr;
}
/*
PetscErrorCode RHSLocalActive(Field **f,Field **u,PetscInt xs,PetscInt xm,PetscInt ys,PetscInt ym,PetscReal Mx,PetscReal My,void *ptr)
{
  //PetscErrorCode  ierr;
  AppCtx          *appctx = (AppCtx*)ptr;
  PetscInt        i,j,sx,sy;
  PetscReal       hx,hy;
  adouble         uc,uxx,uyy,vc,vxx,vyy;
  aField          u_a[ys+ym][xs+xm];           // Independent variables, as an array of structs
  aField          f_a[ys+ym][xs+xm];           // Dependent variables, as an array of structs
  //aField          **u_a,**f_a;

  PetscFunctionBeginUser;

  //ierr = PetscNew(&u_a);CHKERRQ(ierr);
  //ierr = PetscNew(&f_a);CHKERRQ(ierr);

  hx = 2.50/(PetscReal)(Mx);sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My);sy = 1.0/(hy*hy);

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
      uxx         = (-2.0*uc + u_a[j][modulo(i-1,Mx)].u + u_a[j][modulo(i+1,Mx)].u)*sx;
      uyy         = (-2.0*uc + u_a[modulo(j-1,My)][i].u + u_a[modulo(j+1,My)][i].u)*sy;
      vc          = u_a[j][i].v;
      vxx         = (-2.0*vc + u_a[j][modulo(i-1,Mx)].v + u_a[j][modulo(i+1,Mx)].v)*sx;
      vyy         = (-2.0*vc + u_a[modulo(j-1,My)][i].v + u_a[modulo(j+1,My)][i].v)*sy;
      f_a[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);
      f_a[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc;

      // Mark dependence
      f_a[j][i].u >>= f[j][i].u;f_a[j][i].v >>= f[j][i].v;
    }
  }
  trace_off();  // ----------------------------------------------- End of active section

  //ierr = PetscFree(u_a);CHKERRQ(ierr);
  //ierr = PetscFree(f_a);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
*/
PetscErrorCode RHSLocalActive(Field **f,Field **u,PetscInt xs,PetscInt xm,PetscInt ys,PetscInt ym,PetscReal Mx,PetscReal My,void *ptr)
{
  //PetscErrorCode  ierr;
  AppCtx          *appctx = (AppCtx*)ptr;
  PetscInt        i,j,k = 0,sx,sy;
  PetscReal       hx,hy;
  adouble         uc,uxx,uyy,vc,vxx,vyy;
  adouble         *u_a = new adouble[2*(xs+xm)*(ys+ym)];
  adouble         *f_a = new adouble[2*(xs+xm)*(ys+ym)];

  PetscFunctionBeginUser;
  hx = 2.50/(PetscReal)(Mx);sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(My);sy = 1.0/(hy*hy);

  trace_on(1);  // ----------------------------------------------- Start of active section

  // Mark independence
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      u_a[k++] <<= u[j][i].u;u_a[k++] <<= u[j][i].v;
    }
  }
  k = 0;
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {

      // Compute function over the locally owned part of the grid
      uc          = u_a[k++];
      uxx         = (-2.0*uc + u_a[m_map(i-1,j,0,Mx,My,2)] + u_a[m_map(i+1,j,0,Mx,My,2)])*sx;
      uyy         = (-2.0*uc + u_a[m_map(i,j-1,0,Mx,My,2)] + u_a[m_map(i,j+1,0,Mx,My,2)])*sy;
      vc          = u_a[k--];
      vxx         = (-2.0*vc + u_a[m_map(i-1,j,1,Mx,My,2)] + u_a[m_map(i+1,j,1,Mx,My,2)])*sx;
      vyy         = (-2.0*vc + u_a[m_map(i,j-1,1,Mx,My,2)] + u_a[m_map(i,j+1,1,Mx,My,2)])*sy;
      f_a[k++] = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);
      f_a[k--] = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc;

      // Mark dependence
      f_a[k++] >>= f[j][i].u;f_a[k++] >>= f[j][i].v;
    }
  }
  trace_off();  // ----------------------------------------------- End of active section

  delete[] f_a;
  delete[] u_a;

  PetscFunctionReturn(0);
}
PetscErrorCode RHSLocalPassive(Field **f,Field **u,PetscInt xs,PetscInt xm,PetscInt ys,PetscInt ym,PetscReal Mx,PetscReal My,void *ptr)
{
  AppCtx        *appctx = (AppCtx*)ptr;
  PetscInt      i,j,sx,sy;
  PetscReal     hx,hy;
  PetscScalar   uc,uxx,uyy,vc,vxx,vyy;

  PetscFunctionBeginUser;
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
  PetscInt       Mx,My,xs,ys,xm,ym,dofs;
  Field          **u,**f;
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
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  if (!appctx->no_an) {
    RHSLocalActive(f,u,xs,xm,ys,ym,Mx,My,appctx);
  } else {
    RHSLocalPassive(f,u,xs,xm,ys,ym,Mx,My,appctx);
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
  PetscInt       i,j,k = 0,Mx,My,xs,ys,xm,ym,N,dofs,row[1],col[1];
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
    PetscScalar *fz;
    Field       **frhs;

    ierr = PetscMalloc1(N,&fz);CHKERRQ(ierr);
    zos_forward(1,N,N,0,u_vec,fz);

    ierr = PetscMalloc1(N,&frhs);CHKERRQ(ierr);		// FIXME: Memory is not contiguous
    for (j=ys; j<ys+ym; j++) {
      ierr = PetscMalloc1(N,&frhs[j]);CHKERRQ(ierr);
    }
    RHSLocalPassive(frhs,u,xs,xm,ys,ym,Mx,My,appctx);

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
  J = myalloc2(N,N);
  jacobian(1,N,N,u_vec,J); 	// TODO: Use sparse jacobian. Generate sparsity pattern with `jac_pat`
  // FIXME: this probably won't work in parallel
  ierr = PetscFree(u_vec);CHKERRQ(ierr);

  // Insert entries one-by-one
  for(j=0;j<N;j++){
    for(i=0;i<N;i++){
      if(fabs(J[j][i])!=0.){
        row[0] = j; col[0] = i; // TODO: better to insert row-by-row, similarly as with the stencil
        ierr = MatSetValues(A,1,row,1,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  myfree2(J);

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
