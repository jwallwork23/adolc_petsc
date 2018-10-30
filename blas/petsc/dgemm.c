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
  PetscScalar    one = 1.;
  PetscInt*      m = (PetscInt*) M;
  PetscInt*      n = (PetscInt*) N;
  PetscInt       mn = (*m)*(*n);

  PetscFunctionBegin;

  /* Undifferentiated call */
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

  /* Differentiated call */
  ierr = ZeroOut(mn,CD);CHKERRQ(ierr);
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,AD,LDA,B,LDB,BETA,CD,LDC);
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,BD,LDB,&one,CD,LDC);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDGEMMReverse(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AB,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BB,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CB,PetscBLASInt *LDC)
{
  PetscErrorCode ierr;
  PetscScalar    one = 1.;
  PetscInt*      m = (PetscInt*) M;
  PetscInt*      n = (PetscInt*) N;
  PetscInt*      k = (PetscInt*) K;
  PetscInt       mk = (*m)*(*k);
  PetscInt       kn = (*k)*(*n);
  char           leaveA,leaveB;
  char           trans[2] = "NT";

  PetscFunctionBegin;

  if (*TRANSA == trans[0])
    leaveA = trans[1];
  else
    leaveA = trans[0];
  if (*TRANSB == trans[0])
    leaveB = trans[1];
  else
    leaveB = trans[0];

  ierr = ZeroOut(mk,AB);CHKERRQ(ierr);
  ierr = ZeroOut(kn,BB);CHKERRQ(ierr);

  /* Case 1: TRANSA == TRANSB == "N" */
  BLASgemm_(&trans[0],&leaveB,M,K,N,ALPHA,CB,LDC,B,LDB,&one,AB,LDA);
  BLASgemm_(&leaveA,&trans[0],M,N,K,ALPHA,A,LDA,CB,LDC,&one,BB,LDB);
  PetscFunctionReturn(0);
}

