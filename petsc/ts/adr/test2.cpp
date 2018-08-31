#include <petscts.h>
#include <adolc/adolc.h>
#include <iostream>

using namespace std;

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} aField;

PetscErrorCode FieldMallocInner(PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscScalar    *aa;

  PetscFunctionBegin;
  ierr = PetscMalloc1(xm*ym,&aa);CHKERRQ(ierr);
  ierr = PetscMalloc1(ym,a);CHKERRQ(ierr);
  for (i=0; i<ym; i++) (*a)[i] = aa + i*xm - xs;
  *a -= ys;
  ierr = PetscFree(aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FieldMalloc(PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscInt dof,void *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = FieldMallocInner(xs*dof,ys,xm*dof,ym,(PetscScalar***)array);CHKERRQ(ierr);
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
  adouble        tmp;
  aField         **A = NULL;
  //adouble        *aa;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  PetscFunctionBeginUser;

  cout << "Matrix dimension ?= ";
  cin >> n;
  cout << endl;

  Field          **a;
  ierr = FieldMalloc(-1,-1,n+2,n+2,2,&a);CHKERRQ(ierr);
//  ierr = FieldMalloc(0,0,n,n,2,&a);CHKERRQ(ierr);
//  Field a[n][n];


  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      a[i][j].u = i;
      a[i][j].v = j;
    }
  }
  a[-1][n].u = 0.;	// This can only be achieved by using ghost points
  //ierr = PetscMalloc1(n*n,&aa);CHKERRQ(ierr);
/*
  trace_on(1);
  A = new aField*[n];
  for (i=0; i<n; i++) {
    A[i] = new aField[n];
    for (j=0; j<n; j++) {
      A[i][j].u <<= a[i][j].u;A[i][j].v <<= a[i][j].v;

      tmp = A[i][j].u;A[i][j].u = A[i][j].v;A[i][j].v = tmp;

      A[i][j].u >>= a[i][j].u;A[i][j].v >>= a[i][j].v;
    }
  }
  trace_off();

  delete[] A;
*/
  //ierr = PetscFree(aa);CHKERRQ(ierr);

  ierr = FieldFree(-1,-1,n+2,n+2,2,&a);CHKERRQ(ierr);
//  ierr = FieldFree(0,0,n,n,2,&a);CHKERRQ(ierr);

  ierr = PetscFinalize();

  return ierr;
}
