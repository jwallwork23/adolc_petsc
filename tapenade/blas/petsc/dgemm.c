#include <petscblaslapack.h>


PetscErrorCode ZeroOut(PetscInt m,PetscScalar *array)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<m; ++i)
    array[i] = 0.;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDGEMMForward(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AD,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BD,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CD,PetscBLASInt *LDC)
{
  PetscErrorCode ierr;
  PetscScalar    ONE = 1.;
  PetscInt*      m = (PetscInt*) M;
  PetscInt*      n = (PetscInt*) N;
  PetscInt       l = (*m)*(*n);

  PetscFunctionBegin;

  /* Undifferentiated call */
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

  /* Differentiated call */
  ierr = ZeroOut(l,CD);CHKERRQ(ierr);
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,AD,LDA,B,LDB,BETA,CD,LDC);
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,BD,LDB,&ONE,CD,LDC);
  PetscFunctionReturn(0);
}

