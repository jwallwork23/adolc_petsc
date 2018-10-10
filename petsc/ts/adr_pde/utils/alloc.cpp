#include "contexts.cpp"

extern PetscErrorCode AFieldGiveGhostPoints2d(DM da,AField *cgs,AField **a2d[]); // TODO: Generalise
extern PetscErrorCode ConvertTo1Array2d(DM da,Field **u,PetscScalar *u_vec); // TODO:Generalise
extern PetscErrorCode CombineTo1Array2d(DM da,Field **u,Field **v,PetscScalar *u_vec); // TODO: Generalise
extern PetscErrorCode InitialConditions(DM,Vec);


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

/* 
  Convert a 2-array Field defined on a DMDA to a 1-array TODO: Generalise
*/
PetscErrorCode ConvertTo1Array2d(DM da,Field **u,PetscScalar *u_vec)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k = 0,gxs,gys,gxm,gym;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_vec[k++] = u[j][i].u;
      u_vec[k++] = u[j][i].v;
    }
  }
  PetscFunctionReturn(0);
}

/*
  Convert two 2-array Fields defined on a DMDA to a 1-array TODO: Generalise
*/
PetscErrorCode CombineTo1Array2d(DM da,Field **u,Field **v,PetscScalar *u_vec)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k = 0,gxs,gys,gxm,gym;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_vec[k++] = u[j][i].u;
      u_vec[k++] = u[j][i].v;
    }
  }
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++) {
      u_vec[k++] = v[j][i].u;
      u_vec[k++] = v[j][i].v;
    }
  }
  PetscFunctionReturn(0);
}

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

