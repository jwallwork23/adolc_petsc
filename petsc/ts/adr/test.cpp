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
  aField          **A = NULL,**B = NULL;
  aField          *ad = NULL,*bd = NULL;
  PetscInt        ls=0,lm=3,i,j,dof=2,gs,gm;
  adouble         tmp;
  PetscBool       ghost=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-ghost",&ghost,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&lm,NULL);CHKERRQ(ierr);
  gs=ls-1,gm=lm+2;
  Field           aa[lm][lm],bb[lm][lm];

  //ierr = aFieldMalloc(gs,gs,gm,gm,2,&a);CHKERRQ(ierr);

  printf("Before:\n\n");
  for (j=ls; j<lm; j++) {
    for (i=ls; i<lm; i++) {
      aa[i][j].u = i;
      aa[i][j].v = j;
      printf("%f, %f\n",aa[i][j].u,aa[i][j].v);
    }
  }


  // Allocate memory for 2-array aFields
  if (ghost) {
    ad = new aField[gm*gm];
    A = new aField*[gm];
    for (i=gs; i<gm; i++) {
      A[i] = new aField[gm];
      A[i] = ad + dof*i*gm - dof*gs;
    }
    A -= gs;

    bd = new aField[gm*gm];
    B = new aField*[gm];
    for (i=gs; i<gm; i++) {
      B[i] = new aField[gm];
      B[i] = bd + dof*i*gm - dof*gs;
    }
    B -= gs;
   } else {
    ad = new aField[lm*lm];
    A = new aField*[lm];
    for (i=ls; i<lm; i++) {
      A[i] = new aField[lm];
      A[i] = ad + i*lm - ls;
    }
    A -= ls;

    bd = new aField[lm*lm];
    B = new aField*[lm];
    for (i=ls; i<lm; i++) {
      B[i] = new aField[lm];
      B[i] = bd + i*lm - ls;
    }
    B -= ls;
  }

  // Perform active section
  trace_on(1);
  for (i=ls; i<lm; i++) {
    for (j=ls; j<lm; j++) {
      //printf("i=%d,j=%d\n",i,j);
      //printf("  %d\n",&A[i][j]);
      //printf("    %d\n",&A[i][j].v);
      A[i][j].u <<= aa[i][j].u;
      A[i][j].v <<= aa[i][j].v;
    }
  }
  if (!ghost) {
    ls += 1;
  }
  for (i=ls; i<lm; i++) {
    for (j=ls; j<lm; j++) {
      B[i][j].u = A[i-1][j].u;
      B[i][j].v = A[i][j-1].v;
    }
  }

  if (!ghost) {
    ls -= 1;
  }
  for (i=ls; i<lm; i++) {
    for (j=ls; j<lm; j++) {
      B[i][j].u >>= bb[i][j].u;
      B[i][j].v >>= bb[i][j].v;
    }
  }
  trace_off();

  printf("\nAfter:\n\n");
  for (j=ls; j<lm; j++) {
    for (i=ls; i<lm; i++) {
      printf("%f, %f\n",bb[i][j].u,bb[i][j].v);
    }
  }

  printf("\nDone. Now need to destroy and deallocate aField.\n");
  delete[] A;
  delete[] ad;
  printf("Done.\n");

  //ierr = aFieldFree(gs,gs,gm,gm,2,&a);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
