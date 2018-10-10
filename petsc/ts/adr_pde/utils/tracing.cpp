#include "alloc.cpp"

#define tag 1

extern PetscErrorCode IFunctionLocalPassive(DMDALocalInfo*,PetscReal,Field**,Field**,Field**,void*);
extern PetscErrorCode IFunctionLocalActive(DMDALocalInfo*,PetscReal,Field**,Field**,Field**,void*);
extern PetscErrorCode IFunction(TS,PetscReal,Vec,Vec,Vec,void*);


PetscErrorCode IFunctionLocalPassive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,sx,sy;
  PetscScalar    uc,uxx,uyy,vc,vxx,vyy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  hx = 2.50/(PetscReal)(info->mx); sx = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info->my); sy = 1.0/(hy*hy);

  /* Get local grid boundaries */
  xs = info->xs; xm = info->xm; ys = info->ys; ym = info->ym;

  /* Compute function over the locally owned part of the grid */
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

PetscErrorCode IFunctionLocalActive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal      hx,hy,sx,sy;
  adouble        uc,uxx,uyy,vc,vxx,vyy;
  PetscErrorCode ierr;
  AField         **f_a = appctx->f_a,**u_a = appctx->u_a;
  PetscScalar    dummy;

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
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      if ((i < xs) || (i >= xs+xm) || (j < ys) || (j >= ys+ym)) {
        f_a[j][i].u >>= dummy;
        f_a[j][i].v >>= dummy;
      } else {
        f_a[j][i].u >>= f[j][i].u;
        f_a[j][i].v >>= f[j][i].v;
      }
    }
  }
  trace_off();  // ----------------------------------------------- End of active section
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IFunction(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  DMDALocalInfo  info;
  PetscErrorCode ierr;
  Field          **u,**f,**udot;
  Vec            localU;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
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
  ierr = DMDAVecGetArrayRead(da,Udot,&udot);CHKERRQ(ierr);

  if (!appctx->adctx->no_an) {
    ierr = IFunctionLocalActive(&info,ftime,u,udot,f,appctx);CHKERRQ(ierr);
  } else {
    ierr = IFunctionLocalPassive(&info,ftime,u,udot,f,appctx);CHKERRQ(ierr);
  }

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,Udot,&udot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

