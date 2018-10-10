#include "tracing.cpp"

/* Implicit timestepping */
extern PetscErrorCode IJacobianLocalByHand(DMDALocalInfo*,PetscReal,Field**,Field**,PetscReal,Mat,Mat,void*);
extern PetscErrorCode IJacobianLocalAdolc(DMDALocalInfo*,PetscReal,Field**,Field**,PetscReal,Mat,Mat,void*);
extern PetscErrorCode IJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void*);
extern PetscErrorCode RHSJacobianByHand(TS,PetscReal,Vec,Mat,Mat,void*);

/* Explicit timestepping */
extern PetscErrorCode RHSJacobianByHand(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSJacobianADOLC(TS,PetscReal,Vec,Mat,Mat,void*);


PetscErrorCode IJacobianLocalByHand(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,PetscReal a,Mat A,Mat B,void *ptr)
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

PetscErrorCode IJacobianLocalAdolc(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,PetscReal a,Mat A,Mat B,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscScalar    *u_vec;

  PetscFunctionBegin;

  /*
    Convert array of structs to a 2-array, so this can be read by ADOL-C

    TODO: Generalise and then put these allocations into Jacobian computation function
  */
  ierr = PetscMalloc1(appctx->adctx->n,&u_vec);CHKERRQ(ierr);
  ierr = ConvertTo1Array2d(info->da,u,u_vec);CHKERRQ(ierr);

  //ierr = PetscMalloc1(2*appctx->adctx->n,&u_vec);CHKERRQ(ierr);
  //ierr = CombineTo1Array2d(info->da,u,udot,u_vec);CHKERRQ(ierr);

  /*
    For an implicit Jacobian we may use the rule that
       G = M*xdot - f(x)    ==>     dG/dx = a*M - df/dx,
    where a = d(xdot)/dx is a constant.

    TODO: Combine in AdolcComputeIJacobian

    First, calculate the -df/dx part using ADOL-C...
  */
  ierr = AdolcComputeRHSJacobian(A,u_vec,appctx->adctx);CHKERRQ(ierr);
  ierr = PetscFree(u_vec);CHKERRQ(ierr);

  /*
    Assemble local matrix
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(A,a);     /* Add a*M TODO: see above */
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobianMatFree(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A_shell,Mat B,void *ctx)
{
  MatAppCtx            *mctx;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A_shell,(void **)&mctx);CHKERRQ(ierr);

  mctx->time  = t;
  mctx->shift = a;
  if (mctx->ts != ts) mctx->ts = ts;
  if (mctx->actx != ctx) mctx->actx = (AppCtx*) ctx;
  ierr = VecCopy(X,mctx->X);CHKERRQ(ierr);
  ierr = VecCopy(Xdot,mctx->Xdot);CHKERRQ(ierr);
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


PetscErrorCode RHSJacobianADOLC(TS ts,PetscReal t,Vec U,Mat A,Mat BB,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  DM             da;
  PetscErrorCode ierr;
  PetscScalar    *u_vec;
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

  /* Convert array of structs to a 1-array, so this can be read by ADOL-C */
  ierr = PetscMalloc1(appctx->adctx->n,&u_vec);CHKERRQ(ierr);
  ierr = ConvertTo1Array2d(da,u,u_vec);CHKERRQ(ierr);

  /*
    Compute Jacobian using ADOL-C
  */
  ierr = AdolcComputeRHSJacobian(A,u_vec,appctx->adctx);CHKERRQ(ierr);
  ierr = PetscFree(u_vec);CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localU);CHKERRQ(ierr);

  /*
    Assemble local matrix
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

