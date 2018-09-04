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
/*
PetscErrorCode aFieldAllocInner(aField *a,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,adouble **A[])
{
  PetscInt       j;

  PetscFunctionBegin;
  for (j=0; j<ym; j++)
    A[j] = a + j*xm - xs;
  A -= ys;

  PetscFunctionReturn(0);
}

PetscErrorCode aFieldAlloc(aField *a,PetscInt xs,PetscInt ys,PetscInt xm,PetscInt ym,PetscInt dof,void *array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = aFieldAllocInner(a,xs*dof,ys,xm*dof,ym,(adouble***)array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  aField          **A = NULL,**B = NULL;
  aField          *ad = NULL,*bd = NULL;
  PetscInt        xs=0,ys=0,xss=0,yss=0,xm=3,ym=3,i,j,dof=2,gxs,gys,gxm,gym;
  adouble         tmp;
  PetscBool       ghost=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(NULL,NULL,"-ghost",&ghost,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-xm",&xm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-ym",&ym,NULL);CHKERRQ(ierr);
  gxs=xs-1,gys=ys-1,gxm=xm+2,gym=ym+2;
  Field           aa[ym][xm],bb[ym][xm];

  printf("Before:\n");
  printf("\nu component:\n");
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      aa[j][i].u = i+1;
      printf("%f, ",aa[j][i].u);
    }
    printf("\n");
  }
  printf("\nv component:\n");
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      aa[j][i].v = j+1;
      printf("%f, ",aa[j][i].v);
    }
    printf("\n");
  }


  // TODO: create functions to wrap these allocations in
  if (ghost) {
    ad = new aField[dof*gxm*gym];
    A = new aField*[gym];
    for (j=0; j<gym; j++) {
      A[j] = new aField[gxm];
      A[j] = ad + dof*j*gxm - dof*gxs;
    }
    A -= gys;

    bd = new aField[dof*gxm*gym];
    B = new aField*[gym];
    for (j=0; j<gym; j++) {
      B[j] = new aField[gxm];
      B[j] = bd + dof*j*gxm - dof*gxs;
    }
    B -= gys;

   } else {

    ad = new aField[dof*xm*ym];
    A = new aField*[ym];
    for (j=0; j<ym; j++) {
      A[j] = new aField[xm];
      A[j] = ad + dof*j*xm - dof*xs;
    }
    A -= ys;

    bd = new aField[dof*xm*ym];
    B = new aField*[ym];
    for (j=0; j<ym; j++) {
      B[j] = new aField[xm];
      B[j] = bd + dof*j*xm - dof*xs;
    }
    B -= ys;
  }

  // Perform active section
  trace_on(1);
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      //printf("i=%d,j=%d\n",i,j);
      //printf("  %d\n",&A[j][i]);
      //printf("    %d\n",&A[j][i].v);
      A[j][i].u <<= aa[j][i].u;
      A[j][i].v <<= aa[j][i].v;
    }
  }
  if (!ghost) {
    xss += 1;yss += 1;
  }
  for (j=ys; j<ym; j++) {
    for (i=xss; i<xm; i++) {
      B[j][i].u = A[j][i-1].u;
    }
  }
  for (j=yss; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      B[j][i].v = A[j-1][i].v;
    }
  }
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      B[j][i].u >>= bb[j][i].u;
      B[j][i].v >>= bb[j][i].v;
    }
  }
  trace_off();

  printf("\nAfter:\n");
  printf("\nu component:\n");
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      printf("%f, ",bb[j][i].u);
    }
    printf("\n");
  }
  printf("\nv component:\n");
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      printf("%f, ",bb[j][i].v);
    }
    printf("\n");
  }

/*
  A += ys;
  B += ys;
  for (j=ys; j<ym; j++){
    A[j] += dof*xs - dof*j*xm;
    B[j] += dof*xs - dof*j*xm;
  }
*/

  printf("\nDone. Now need to destroy and deallocate aField.\n");
  delete[] B;
  printf("Freed B\n");
  delete[] bd;
  printf("Freed bd\n");
  delete[] A;
  printf("Freed A\n");
  delete[] ad;
  printf("Done.\n");

  ierr = PetscFinalize();
  return ierr;
}
