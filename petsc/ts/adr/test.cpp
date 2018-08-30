#include <petscts.h>
#include <adolc/adolc.h>
#include "utils.c"

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} aField;

PetscErrorCode VecSetMemory(PetscScalar *aa,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscMalloc1(ym,a);CHKERRQ(ierr);
  for (i=0; i<ym; i++) (*a)[i] = aa + i*xm - xs;
  *a -= ys;
  PetscFunctionReturn(0);
}

PetscErrorCode VecFreeMemory(PetscScalar *aa,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscScalar **a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  dummy = (void*)(*a+ys);
  ierr = PetscFree(dummy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMDASetMemory(PetscScalar *aa,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscInt dof,void *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSetMemory(aa,xs*dof,ys,xm*dof,ym,(PetscScalar***)array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMDAFreeMemory(PetscScalar *aa,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscInt dof,void *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecFreeMemory(aa,xs*dof,ys,xm*dof,ym,(PetscScalar***)array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  Field           **a;
  PetscInt        i,j;
  PetscScalar     *aa;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  PetscFunctionBeginUser;

  ierr = PetscMalloc1(8,&aa);
  for(i=0;i<8;i++){
    aa[i]=i;
  }
  ierr = DMDASetMemory(aa,0,0,2,2,2,&a);CHKERRQ(ierr);

  for (j=0; j<2; j++){
    for (i=0; i<2; i++){
      printf("%f, %f\n",a[j][i].u,a[j][i].v);
    }
  }

  ierr = DMDAFreeMemory(aa,0,0,2,2,2,&a);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
