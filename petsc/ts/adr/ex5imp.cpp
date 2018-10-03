static char help[] = "Demonstrates automatic Jacobian generation using ADOL-C for a time-dependent PDE in 2d, solved using implicit timestepping.\n";

/*
  See ex5.c for details on the equation.

  Here implicit Crank-Nicolson timestepping is used to solve the same problem as in ex5.c. Another
  key difference is that functions and Jacobians may optionally be calculated in a local sense. The
  the local implementations IFunctionLocal and IJacobianLocal are passed to the TS solver using
  DMTSSetIFunctionLocal and DMTSSetIJacobianLocal.

  Credit for this implementation to Hong Zhang.
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>	// Include ADOL-C

#define tag 1

/* (Passive) field for two DOFs */
typedef struct {
  PetscScalar u,v;
} Field;

/* Active field for two DOFs */
typedef struct {
  adouble u,v;
} AField;

/* Application context */
typedef struct {
  PetscReal   D1,D2,gamma,kappa;
  PetscBool   no_an;
  AField      **u_a,**f_a;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(DM,Vec);
static PetscErrorCode IFunctionLocalPassive(DMDALocalInfo*,PetscReal,Field**,Field**,Field**,void*);
static PetscErrorCode IFunctionLocalActive(DMDALocalInfo*,PetscReal,Field**,Field**,Field**,void*);
static PetscErrorCode IJacobianLocalByHand(DMDALocalInfo*,PetscReal,Field**,Field**,PetscReal,Mat,Mat,void*);
static PetscErrorCode IJacobianLocalAdolc(DMDALocalInfo*,PetscReal,Field**,Field**,PetscReal,Mat,Mat,void*);
extern PetscErrorCode AFieldGiveGhostPoints2d(DM da,AField *cgs,AField **a2d[]);

int main(int argc,char **argv)
{
  TS             ts;                  /* ODE integrator */
  Vec            x;                   /* solution */
  PetscErrorCode ierr;
  DM             da;
  AppCtx         appctx;
  PetscInt       gxs,gys,gxm,gym,dof;
  AField         **u_a = NULL,**f_a = NULL,*u_c = NULL,*f_c = NULL;
  PetscBool      byhand = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
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
  ierr = VecGetSize(x,&dof);CHKERRQ(ierr);

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
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  if (!appctx.no_an) {
    ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)IFunctionLocalActive,&appctx);CHKERRQ(ierr);
  }
  else {
    ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)IFunctionLocalPassive,&appctx);CHKERRQ(ierr);
  }

  if (!byhand) {
    ierr = DMDATSSetIJacobianLocal(da,(DMDATSIJacobianLocal)IJacobianLocalAdolc,&appctx);CHKERRQ(ierr);
  } else {
    ierr = DMDATSSetIJacobianLocal(da,(DMDATSIJacobianLocal)IJacobianLocalByHand,&appctx);CHKERRQ(ierr);
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
  ierr = TSSetTimeStep(ts,10);CHKERRQ(ierr);
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

static PetscErrorCode IFunctionLocalPassive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  hx = 2.50/(PetscReal)(info->mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info->my); sy = 1.0/(hy*hy);

  /*
     Get local grid boundaries
  */
  xs = info->xs; xm = info->xm;
  ys = info->ys; ym = info->ym;
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
      f[j][i].u = udot[j][i].u - appctx->D1*(uxx + uyy) + uc*vc*vc - appctx->gamma*(1.0 - uc);
      f[j][i].v = udot[j][i].v - appctx->D2*(vxx + vyy) - uc*vc*vc + (appctx->gamma + appctx->kappa)*vc;
    }
  }
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunctionLocalActive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal      hx,hy,sx,sy;
  adouble        uc,uxx,uyy,vc,vxx,vyy;
  PetscErrorCode ierr;
  AField         **f_a = appctx->f_a,**u_a = appctx->u_a;

  PetscFunctionBegin;
  hx = 2.50/(PetscReal)(info->mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info->my); sy = 1.0/(hy*hy);
  xs = info->xs; xm = info->xm; gxs = info->gxs; gxm = info->gxm;
  ys = info->ys; ym = info->ym; gys = info->gys; gym = info->gym;

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
  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u_a[j][i].u;
      uxx       = (-2.0*uc + u_a[j][i-1].u + u_a[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u_a[j-1][i].u + u_a[j+1][i].u)*sy;
      vc        = u_a[j][i].v;
      vxx       = (-2.0*vc + u_a[j][i-1].v + u_a[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u_a[j-1][i].v + u_a[j+1][i].v)*sy;
      f_a[j][i].u = udot[j][i].u - appctx->D1*(uxx + uyy) + uc*vc*vc - appctx->gamma*(1.0 - uc);
      f_a[j][i].v = udot[j][i].v - appctx->D2*(vxx + vyy) - uc*vc*vc + (appctx->gamma + appctx->kappa)*vc;
    }
  }

  /*
    Mark dependence

    NOTE: Ghost points are marked as dependent in order to vastly simplify index notation
          during Jacobian assembly.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      f_a[j][i].u >>= f[j][i].u;	// FIXME: f does not have ghost points
      f_a[j][i].v >>= f[j][i].v;
    }
  }
  trace_off();  // ----------------------------------------------- End of active section
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobianLocalByHand(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,PetscReal a,Mat A,Mat B,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;     /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,vc;
  MatStencil     stencil[6],rowstencil;
  PetscScalar    entries[6];

  PetscFunctionBegin;
  hx = 2.50/(PetscReal)(info->mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info->my); sy = 1.0/(hy*hy);
  xs = info->xs; xm = info->xm;
  ys = info->ys; ym = info->ym;

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

  ierr = PetscLogFlops(19*xm*ym);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobianLocalAdolc(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,PetscReal a,Mat A,Mat B,void *ptr)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k = 0,xm,ym,gxs,gxm,gys,gym,m,n,dofs = 2;
  PetscScalar    *u_vec,**J;

  PetscFunctionBegin;
  xm = info->xm; gxs = info->gxs; gxm = info->gxm;
  ym = info->ym; gys = info->gys; gym = info->gym;

  /* Convert array of structs to a 2-array, so this can be read by ADOL-C */
  m = dofs*gxm*gym;  // Number of dependent variables
  n = m;             // Number of independent variables
  ierr = PetscMalloc1(n,&u_vec);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_vec[k++] = u[j][i].u;
      u_vec[k++] = u[j][i].v;
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

  /*
    Assemble local matrix
  */
  ierr = PetscLogFlops(19*xm*ym);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /*
    Next, assemble a*M
  */
  for (i=0; i<m; i++) {
    ierr = MatSetValuesLocal(A,1,&i,1,&i,&a,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Shift indices in AField to endow it with ghost points.  // TODO: Generalise
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
