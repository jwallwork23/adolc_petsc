#include "init.cpp"

PetscErrorCode IFunctionLocalPassive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      hx,hy,s2x,s2y,s1x,s1y;
  PetscScalar    uc,ux,uxx,uy,uyy,vc,vx,vxx,vy,vyy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  hx = 2.50/(PetscReal)(info->mx); s1x = 1.0/(2.0*hx); s2x = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info->my); s1y = 1.0/(2.0*hy); s2y = 1.0/(hy*hy);

  /* Get local grid boundaries */
  xs = info->xs; xm = info->xm; ys = info->ys; ym = info->ym;

  /* Compute function over the locally owned part of the grid */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      uc        = u[j][i].u;
      ux        = (u[j][i+1].u - u[j][i-1].u)*s1x;
      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*s2x;
      uy        = (u[j+1][i].u - u[j-1][i].u)*s1y;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*s2y;

      vc        = u[j][i].v;
      vx        = (u[j][i+1].v - u[j][i-1].v)*s1x;
      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*s2x;
      vy        = (u[j+1][i].v - u[j-1][i].v)*s1y;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*s2y;
      f[j][i].u = udot[j][i].u - appctx->D*(uxx + uyy) + appctx->kappa*(uc*ux+vc*uy);
      f[j][i].v = udot[j][i].v - appctx->D*(vxx + vyy) + appctx->kappa*(uc*vx+vc*vy);
    }
  }
  ierr = PetscLogFlops(16*xm*ym);CHKERRQ(ierr); // FIXME
  PetscFunctionReturn(0);
}

PetscErrorCode IFunctionLocalActive(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal      hx,hy,s1x,s1y,s2x,s2y;
  adouble        uc,ux,uxx,uy,uyy,vc,vx,vxx,vy,vyy;
  PetscErrorCode ierr;
  AField         **f_a = appctx->f_a,**u_a = appctx->u_a;
  PetscScalar    dummy;

  PetscFunctionBegin;
  hx = 2.50/(PetscReal)(info->mx); s1x = 1.0/(2.0*hx); s2x = 1.0/(hx*hx);
  hy = 2.50/(PetscReal)(info->my); s1y = 1.0/(2.0*hy); s2y = 1.0/(hy*hy);
  xs = info->xs; xm = info->xm; gxs = info->gxs; gxm = info->gxm;
  ys = info->ys; ym = info->ym; gys = info->gys; gym = info->gym;

  trace_on(1);  // ----------------------------------------------- Start of active section

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
      ux        = (u_a[j][i+1].u - u_a[j][i-1].u)*s1x;
      uxx       = (-2.0*uc + u_a[j][i-1].u + u_a[j][i+1].u)*s2x;
      uy        = (u_a[j+1][i].u - u_a[j-1][i].u)*s1y;
      uyy       = (-2.0*uc + u_a[j-1][i].u + u_a[j+1][i].u)*s2y;

      vc        = u_a[j][i].v;
      vx        = (u_a[j][i+1].v - u_a[j][i-1].v)*s1x;
      vxx       = (-2.0*vc + u_a[j][i-1].v + u_a[j][i+1].v)*s2x;
      vy        = (u_a[j+1][i].v - u_a[j-1][i].v)*s1y;
      vyy       = (-2.0*vc + u_a[j-1][i].v + u_a[j+1][i].v)*s2y;
      f_a[j][i].u = - appctx->D*(uxx + uyy) + appctx->kappa*(uc*ux+vc*uy);
      f_a[j][i].v = - appctx->D*(vxx + vyy) + appctx->kappa*(uc*vx+vc*vy);
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

PetscErrorCode IFunctionLocalActive2(DMDALocalInfo *info,PetscReal t,Field**u,Field**udot,Field**f,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  PetscInt       i,j,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscErrorCode ierr;
  AField         **f_a = appctx->f_a,**udot_a = appctx->udot_a;
  PetscScalar    dummy;

  PetscFunctionBegin;
  xs = info->xs; xm = info->xm; gxs = info->gxs; gxm = info->gxm;
  ys = info->ys; ym = info->ym; gys = info->gys; gym = info->gym;

  trace_on(2);  // ----------------------------------------------- Start of active section

  /*
    Mark independence

    NOTE: Ghost points are marked as independent, in place of the points they represent on
          other processors / on other boundaries.
  */
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      udot_a[j][i].u <<= udot[j][i].u;
      udot_a[j][i].v <<= udot[j][i].v;
    }
  }

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      f_a[j][i].u = udot_a[j][i].u;
      f_a[j][i].v = udot_a[j][i].v;
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

PetscErrorCode IFunctionActive(TS ts,PetscReal ftime,Vec U,Vec Udot,Vec F,void *ptr)
{
  AppCtx         *appctx = (AppCtx*)ptr;
  DM             da;
  DMDALocalInfo  info;
  PetscErrorCode ierr;
  Field          **u,**f,**udot;
  Vec            localU,localUdot;

  PetscFunctionBegin;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localU);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localUdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,U,INSERT_VALUES,localU);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,Udot,INSERT_VALUES,localUdot);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,Udot,INSERT_VALUES,localUdot);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(da,localUdot,&udot);CHKERRQ(ierr);

  /*
    Call local versions

    TODO: This isn't actually evaluating correctly, although it does trace as required. Second
          call should increment first.
  */
  ierr = IFunctionLocalActive(&info,ftime,u,udot,f,appctx);CHKERRQ(ierr);
  ierr = IFunctionLocalActive2(&info,ftime,u,udot,f,appctx);CHKERRQ(ierr);
  
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localU,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,localUdot,&udot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

