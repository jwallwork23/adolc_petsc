#include <petscts.h>
#include <adolc/adolc.h>

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} aField;
/*
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
*/
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  aField          **A = NULL;
  aField          *ad = NULL;
  PetscInt        ls=0,lm=3,i,j,k=0,dof=2,gs,gm;
  PetscScalar     *aa,c=0.;
  adouble         tmp;
  PetscBool       ghost=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-ghost",&ghost,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&lm,NULL);CHKERRQ(ierr);
  gs=ls-1,gm=lm+2;

  //ierr = aFieldMalloc(gs,gs,gm,gm,2,&a);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*lm*lm,&aa);CHKERRQ(ierr);

  for (j=ls; j<lm; j++) {
    for (i=ls; i<lm; i++) {
      aa[k] = k;k++;
      aa[k] = k;k++;
    }
  }


  // Allocate memory for 2-array aField
  if (ghost) {
    ad = new aField[gm*gm];
    A = new aField*[gm];
    for (i=gs; i<gm; i++) {
      A[i] = new aField[gm];
      A[i] = ad + dof*i*gm - dof*gs;
    }
    A -= gs;
   } else {
    ad = new aField[lm*lm];
    A = new aField*[lm];
    for (i=ls; i<lm; i++) {
      A[i] = new aField[lm];
      A[i] = ad + i*lm - ls;
    }
    A -= ls;
  }

  // Perform active section
  k = 0;
  trace_on(1);
  if (ghost) {
    A[-1][0].u <<= c;
  }
  for (i=ls; i<lm; i++) {
    for (j=ls; j<lm; j++) {
      printf("i=%d,j=%d\n",i,j);
      printf("  %d\n",&A[i][j]);
      printf("    %d\n",&A[i][j].v);
      A[i][j].u <<= aa[k++];
      A[i][j].v <<= aa[k--];

      tmp = A[i][j].u;A[i][j].u = A[i][j].v;A[i][j].v = tmp;

      A[i][j].u >>= aa[k++];
      A[i][j].v >>= aa[k++];
    }
  }
  if (ghost) {
    A[-1][0].u >>= c;
  }
  trace_off();

  printf("Done. Now need to destroy and deallocate aField.\n");
  delete[] A;
  delete[] ad;
  printf("Done.\n");

  ierr = PetscFree(aa);CHKERRQ(ierr);
  //ierr = aFieldFree(gs,gs,gm,gm,2,&a);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
