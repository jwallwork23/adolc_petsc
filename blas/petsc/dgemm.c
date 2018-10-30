#include <petscblaslapack.h>


PetscErrorCode ZeroOut(PetscInt m,PetscScalar *array)
{
  PetscInt i;

  PetscFunctionBegin;
  for (i=0; i<m; ++i)
    array[i] = 0.;
  PetscFunctionReturn(0);
}

PetscErrorCode Trans(PetscInt m,PetscInt n,PetscScalar *A,PetscScalar *At)
{
  PetscInt i,j;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      At[j+i*m] = A[i+j*n];
    }
  }
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
  PetscScalar    *ABT,*BBT;

  PetscFunctionBegin;
  ierr = PetscMalloc2(&mk,&ABT,&kn,&BBT);CHKERRQ(ierr);
  if (*TRANSA == trans[0])
    leaveA = trans[1];
  else
    leaveA = trans[0];
  if (*TRANSB == trans[0])
    leaveB = trans[1];
  else
    leaveB = trans[0];

  ierr = ZeroOut(mk,ABT);CHKERRQ(ierr);
  ierr = ZeroOut(kn,BBT);CHKERRQ(ierr);

  /* Case 1: TRANSA == TRANSB == "N" */
  BLASgemm_(&trans[0],&leaveB,M,K,N,ALPHA,CB,LDC,B,LDB,&one,ABT,LDA);
  BLASgemm_(&leaveA,&trans[0],M,N,K,ALPHA,A,LDA,CB,LDC,&one,BBT,LDB);
/*
  if (*TRANSA == trans[0])
    AB = ABT;
  else
    ierr = Trans(m,k,AB,ABT);CHKERRQ(ierr);
  if (*TRANSB == trans[0])
    BB = BBT;
  else
    ierr = Trans(m,k,BB,BBT);CHKERRQ(ierr);
*/
  ierr = PetscFree2(ABT,BBT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

