#include <petscts.h>
#include <iostream>

using namespace std;

typedef struct {
  PetscScalar u,v;
} Field;

PetscErrorCode FieldMallocInner(PetscScalar *aa,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscMalloc1(xm*ym,&aa);CHKERRQ(ierr);
  ierr = PetscMalloc1(ym,a);CHKERRQ(ierr);
  for (i=0; i<ym; i++) (*a)[i] = aa + i*xm - xs;
  *a -= ys;
  PetscFunctionReturn(0);
}

PetscErrorCode FieldMalloc(PetscScalar *aa,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscInt dof,void *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FieldMallocInner(aa,xs*dof,ys,xm*dof,ym,(PetscScalar***)array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FieldFreeInner(PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscScalar **a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  dummy = (void*)(*a+ys);
  ierr = PetscFree(dummy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FieldFree(PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscInt dof,void *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FieldFreeInner(xs*dof,ys,xm*dof,ym,(PetscScalar***)array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n,i,j;
  Field          **a;
  PetscScalar    *aa;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  PetscFunctionBeginUser;

  cout << "Matrix dimension ?= ";
  cin >> n;
  cout << endl;

  ierr = PetscMalloc1(2*(n+1)*(n+1),&aa);CHKERRQ(ierr);
  ierr = FieldMalloc(aa,-1,-1,n+2,n+2,2,a);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      a[i][j].u = i;
      a[i][j].v = j;
    }
  }
  a[-1][n].u = 0.;	// This can only be achieved by using ghost points

  ierr = FieldFree(-1,-1,n+2,n+2,2,&a);CHKERRQ(ierr);
  ierr = PetscFree(aa);CHKERRQ(ierr);

  ierr = PetscFinalize();

  return ierr;
}
