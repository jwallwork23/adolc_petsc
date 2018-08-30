#include <petscts.h>
#include <adolc/adolc.h>

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} aField;

// TODO: From the ADOL-C manual: "DO NOT use malloc() or related C memory allocating functions when declaring adoubles

PetscErrorCode aFieldMallocInner(PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,adouble **a[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  adouble        *aa;

  PetscFunctionBegin;
  ierr = PetscMalloc1(xm*ym,&aa);CHKERRQ(ierr);
  ierr = PetscMalloc1(ym,a);CHKERRQ(ierr);
  for (i=0; i<ym; i++) (*a)[i] = aa + i*xm - xs;
  *a -= ys;
  ierr = PetscFree(aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode aFieldMalloc(PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscInt dof,void *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = aFieldMallocInner(xs*dof,ys,xm*dof,ym,(adouble***)array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode aFieldFreeInner(PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,adouble **a[])
{
  PetscErrorCode ierr;
  void           *dummy;

  PetscFunctionBegin;
  dummy = (void*)(*a+ys);
  ierr = PetscFree(dummy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode aFieldFree(PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscInt dof,void *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = aFieldFreeInner(xs*dof,ys,xm*dof,ym,(adouble***)array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  aField          **a;
  PetscInt        i,j,k=0;
  PetscScalar     *aa;
/*
  PetscScalar     test1;
  adouble         test2;
  double          test3;
*/
  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  PetscFunctionBegin;

  ierr = aFieldMalloc(-1,-1,4,4,2,&a);CHKERRQ(ierr);
  ierr = PetscMalloc1(8,&aa);CHKERRQ(ierr);

  for (j=0; j<2; j++) {
    for (i=0; i<2; i++) {
      aa[k] = k;k++;
      aa[k] = k;k++;
    }
  }
/*
  printf("Size of PetscScalar = %d\n",sizeof(test1));
  printf("Size of adouble     = %d\n",sizeof(test2));
  printf("Size of double      = %d\n",sizeof(test3));
*/

  trace_on(1);
  a[0][0].u <<= aa[0];
  trace_off();

  ierr = PetscFree(aa);CHKERRQ(ierr);
  ierr = aFieldFree(-1,-1,4,4,2,&a);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
