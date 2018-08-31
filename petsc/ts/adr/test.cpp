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

PetscErrorCode aFieldAlloc(aField *a,PetscInt m,PetscInt s,PetscInt dof,aField **A)
{
  PetscInt       i;

  PetscFunctionBegin;
  A = new aField*[m];
  for (i=s; i<m; i++) {
    A[i] = new aField[m];
    A[i] = a + dof*i*m - dof*s;
  }
  A -= s;

  PetscFunctionReturn(0);
}

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

  //ierr = aFieldMalloc(gs,gs,gm,gm,2,&a);CHKERRQ(ierr);

  printf("Before:\n\n");
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      aa[j][i].u = i+1;
      aa[j][i].v = j+1;
      printf("%f, %f\n",aa[j][i].u,aa[j][i].v);
    }
  }


  // Allocate memory for 2-array aFields
/*  
  if (ghost) {
    ad = new aField[gm*gm];
    bd = new aField[gm*gm];
    ierr = aFieldAlloc(ad,gm,gs,dof,A);CHKERRQ(ierr);
    ierr = aFieldAlloc(bd,gm,gs,dof,B);CHKERRQ(ierr);
  } else {
    ad = new aField[lm*lm];
    bd = new aField[lm*lm];
    ierr = aFieldAlloc(ad,lm,ls,dof,A);CHKERRQ(ierr);
    ierr = aFieldAlloc(bd,lm,ls,dof,B);CHKERRQ(ierr);
  }
*/

  if (ghost) {
    ad = new aField[dof*gxm*gym];
    A = new aField*[gym];
    for (j=gys; j<gym; j++) {
      A[j] = new aField[gxm];
      A[j] = ad + dof*j*gxm - dof*gxs;
    }
    A -= gys;

    bd = new aField[dof*gxm*gym];
    B = new aField*[gym];
    for (j=gys; j<gym; j++) {
      B[j] = new aField[gxm];
      B[j] = bd + dof*j*gxm - dof*gxs;
    }
    B -= gys;

   } else {


    ad = new aField[dof*xm*ym];
    A = new aField*[ym];
    for (j=ys; j<ym; j++) {
      A[j] = new aField[xm];
      A[j] = ad + dof*j*xm - dof*xs;
    }
    A -= ys;

    bd = new aField[dof*xm*ym];
    B = new aField*[ym];
    for (j=ys; j<ym; j++) {
      B[j] = new aField[xm];
      B[j] = bd + dof*j*xm - dof*xs;
    }
    B -= ys;
  }

  // Perform active section
  trace_on(1);
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      printf("i=%d,j=%d\n",i,j);
      printf("  %d\n",&A[j][i]);
      printf("    %d\n",&A[j][i].v);
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

  printf("\nAfter:\n\n");
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      printf("%f, %f\n",bb[j][i].u,bb[j][i].v);
    }
  }

  printf("\nDone. Now need to destroy and deallocate aField.\n");
  delete[] A;
  delete[] B;
  delete[] ad;
  delete[] bd;
  printf("Done.\n");

  //ierr = aFieldFree(gs,gs,gm,gm,2,&a);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
