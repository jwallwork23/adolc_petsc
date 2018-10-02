#include <petscdm.h>
#include <petscdmda.h>
#include <adolc/adolc.h>


extern PetscErrorCode GiveGhostPoints2d(DM da,void *cgs,void *a2d);
extern PetscErrorCode AdoubleGiveGhostPoints2d(DM da,adouble *cgs,adouble **a2d[]);
extern PetscErrorCode ConvertTo1Array(DM da,PetscScalar **u,PetscScalar *u_vec);
extern PetscErrorCode ConvertTo1Array2d(DM da,PetscScalar **u,PetscScalar *u_vec);


PetscErrorCode GiveGhostPoints2d(DM da,void *cgs,void *a2d)
{

  PetscFunctionBegin;

  // TODO general format accounting for dimensions and dofs

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

PetscErrorCode ConvertTo1Array(DM da,void **u,PetscScalar *u_vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // FIXME and generalise for dimensions and dofs
  ierr = ConvertTo1Array2d(da,(PetscScalar**)u,u_vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ConvertTo1Array2d(DM da,PetscScalar **u,PetscScalar *u_vec)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k = 0,gxs,gys,gxm,gym;

  PetscFunctionBegin;
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,NULL,&gxm,&gym,NULL);CHKERRQ(ierr);
  for (j=gys; j<gys+gym; j++) {
    for (i=gxs; i<gxs+gxm; i++)
      u_vec[k++] = u[j][i];
  }
  PetscFunctionReturn(0);
}
