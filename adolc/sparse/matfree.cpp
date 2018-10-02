#include <petscsnes.h>
#include <adolc/adolc.h>
#include <adolc/adolc_sparse.h>

#define tag 1

extern PetscErrorCode PrintMat(MPI_Comm comm,const char* name,PetscInt n,PetscInt m,PetscScalar **M);
extern PetscErrorCode PassiveEvaluate(PetscScalar *x,PetscScalar *c);
extern PetscErrorCode ActiveEvaluate(adouble *x,adouble *c);
extern PetscErrorCode JacobianVectorProduct(Mat J,Vec U,Vec Action);
extern PetscErrorCode JacobianTransposeVectorProduct(Mat J,Vec X,Vec Action);

PetscErrorCode AdolcMalloc2(PetscInt m,PetscInt n,PetscScalar **A[])
{
  PetscFunctionBegin;
  *A = myalloc2(m,n);
  PetscFunctionReturn(0);
}

PetscErrorCode AdolcFree2(PetscScalar **A)
{
  PetscFunctionBegin;
  myfree2(A);
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  MPI_Comm        comm = MPI_COMM_WORLD;
  PetscInt        n = 6,m = 3,i,j,ix[m];
  PetscScalar     x[n],c[m];
  adouble         xad[n],cad[m];
  Vec             C,Z;
  Mat             J;

  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;

  /* Give values for independent variables */
  for(i=0;i<n;i++)
    x[i] = log(1.0+i);

  /* Trace function c(x) */
  trace_on(tag);
    for(i=0;i<n;i++)
      xad[i] <<= x[i];

    ierr = ActiveEvaluate(xad,cad);CHKERRQ(ierr);

    for(i=0;i<m;i++)
      cad[i] >>= c[i];
  trace_off();

  /* Function evaluation as above */
  ierr = PetscPrintf(comm,"\n Function evaluation by RHS : ");CHKERRQ(ierr);
  for(j=0;j<m;j++) {
    ierr = PetscPrintf(comm," %e ",c[j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);

  /* Trace over ZOS to check function evaluation and enable reverse mode */
  zos_forward(tag,m,n,1,x,c);
  ierr = PetscPrintf(comm,"\n Function evaluation by ZOS : ");CHKERRQ(ierr);
  for(j=0;j<m;j++) {
    ierr = PetscPrintf(comm," %e ",c[j]);CHKERRQ(ierr);
    ix[j] = j;
  }
  ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);

  /* Insert dependent variable values into a Vec */
  ierr = VecCreate(comm,&C);CHKERRQ(ierr);
  ierr = VecSetSizes(C,PETSC_DECIDE,m);CHKERRQ(ierr);
  ierr = VecSetFromOptions(C);CHKERRQ(ierr);
  ierr = VecSetValues(C,m,ix,c,INSERT_VALUES);CHKERRQ(ierr);
  //ierr = VecView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Create matrix free matrix */
  ierr = MatCreateShell(comm,m,n,m,n,NULL,&J);CHKERRQ(ierr);
  // TODO: Set MatMult using forward mode
  ierr = MatShellSetOperation(J,MATOP_MULT_TRANSPOSE,(void(*)(void))JacobianTransposeVectorProduct);CHKERRQ(ierr);

  /* Evaluate Jacobian matrix free */
  ierr = VecCreate(comm,&Z);CHKERRQ(ierr);
  ierr = VecSetSizes(Z,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(Z);CHKERRQ(ierr);
  //ierr = JacobianTransposeVectorProduct(C,m,n,Z);CHKERRQ(ierr);
  ierr = MatMultTranspose(J,C,Z);CHKERRQ(ierr);	// Note: This has been overloaded for matrix J
  ierr = PetscPrintf(comm,"Jacobian transpose vector product:\n");CHKERRQ(ierr);
  ierr = VecView(Z,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&Z);CHKERRQ(ierr);
  ierr = VecDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return ierr;
}

/* Intended to overload MatMult in matrix-free methods */
PetscErrorCode JacobianVectorProduct(Mat J,Vec X,Vec Action)
{
  PetscErrorCode    ierr;
  PetscInt          m,n;
  const PetscScalar *x;
  PetscScalar       *action,*xx,*ff;

  PetscFunctionBegin;

  /* Read data and allocate memory */
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&x);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&xx);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&ff);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&action);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  for (i=0; i<n; i++)
    xx[i] = x[i];	// FIXME: How to avoid this conversion from read only?

  /* Compute action of Jacobian on vector */
  fos_forward(tag,m,n,0,xx,xx,ff,action);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = VecSetValues(Action,1,&i,&ff[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = PetscFree(ff);CHKERRQ(ierr);
  ierr = PetscFree(xx);CHKERRQ(ierr);
  ierr = PetscFree(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Intended to overload MatMultTranspose in matrix-free methods */
PetscErrorCode JacobianTransposeVectorProduct(Mat J,Vec U,Vec Action)
{
  PetscErrorCode    ierr;
  PetscInt          i,m,n;
  const PetscScalar *u;
  PetscScalar       *action,*uu;
  //PetscScalar       *x,*c;

  PetscFunctionBegin;

  // TODO: How to call zos_forward here?

  /* Read data and allocate memory */
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&u);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&uu);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
/*
  ierr = PetscMalloc1(n,&x);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&c);CHKERRQ(ierr);
  zos_forward(tag,m,n,1,x,c);
  ierr = PetscFree(c);CHKERRQ(ierr);
  ierr = PetscFree(x);CHKERRQ(ierr);
*/
  /* Compute action of Jacobian transpose on vector */
  ierr = PetscMalloc1(n,&action);CHKERRQ(ierr);
  for (i=0; i<m; i++)
    uu[i] = u[i];	// FIXME: How to avoid this conversion from read only?
  fos_reverse(tag,m,n,uu,action);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);

  /* Set values in vector */
  for (i=0; i<n; i++) {
    ierr = VecSetValues(Action,1,&i,&action[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = PetscFree(action);CHKERRQ(ierr);
  ierr = PetscFree(uu);CHKERRQ(ierr);
  ierr = PetscFree(u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PrintMat(MPI_Comm comm,const char* name,PetscInt m,PetscInt n,PetscScalar **M)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"%s \n",name);CHKERRQ(ierr);
  for(i=0; i<m ;i++) {
    ierr = PetscPrintf(comm,"\n %d: ",i);CHKERRQ(ierr);
    for(j=0; j<n ;j++)
      ierr = PetscPrintf(comm," %10.4f ", M[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PassiveEvaluate(PetscScalar *x,PetscScalar *c)
{
  PetscFunctionBegin;
  c[0] = 2*x[0]+x[1]-2.0;
  c[0] += PetscCosScalar(x[3])*PetscSinScalar(x[4]);
  c[1] = x[2]*x[2]+x[3]*x[3]-2.0;
  c[2] = 3*x[4]*x[5] - 3.0+PetscSinScalar(x[4]*x[5]);
  PetscFunctionReturn(0);
}

PetscErrorCode ActiveEvaluate(adouble *x,adouble *c)
{
  PetscFunctionBegin;
  c[0] = 2*x[0]+x[1]-2.0;
  c[0] += PetscCosScalar(x[3])*PetscSinScalar(x[4]);
  c[1] = x[2]*x[2]+x[3]*x[3]-2.0;
  c[2] = 3*x[4]*x[5] - 3.0+PetscSinScalar(x[4]*x[5]);
  PetscFunctionReturn(0);
}

