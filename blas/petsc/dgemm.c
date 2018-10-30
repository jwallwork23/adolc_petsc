#include <petscblaslapack.h>


PetscErrorCode PetscDGEMMForward(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AD,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BD,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CD,PetscBLASInt *LDC)
{
  PetscScalar    one = 1.;
  PetscInt       *m = (PetscInt*) M,*n = (PetscInt*) N,i;

  PetscFunctionBegin;

  /* Undifferentiated call */
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

  /* Differentiated call */
  for (i=0; i<(*m)*(*n); ++i) CD[i] = 0.;
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,AD,LDA,B,LDB,BETA,CD,LDC);
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,BD,LDB,&one,CD,LDC);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscDGEMMReverse(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AB,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BB,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CB,PetscBLASInt *LDC)
{
  PetscErrorCode ierr;
  PetscScalar    one = 1.,*ABT,*BBT;
  PetscInt       *m = (PetscInt*) M,*n = (PetscInt*) N,*k = (PetscInt*) K;
  PetscInt       mk = (*m)*(*k),kn = (*k)*(*n),i,j;
  char           leaveA,leaveB,trans[2] = "NT";

  PetscFunctionBegin;
  ierr = PetscMalloc2(mk,&ABT,kn,&BBT);CHKERRQ(ierr);
  if (*TRANSA == trans[0]) {
    leaveA = trans[1];
    for (i=0; i<mk; ++i) AB[i] = 0.;
  } else {
    leaveA = trans[0];
    for (i=0; i<mk; ++i) ABT[i] = 0.;
  }
  if (*TRANSB == trans[0]) {
    leaveB = trans[1];
    for (i=0; i<kn; ++i) BB[i] = 0.;
  } else {
    leaveB = trans[0];
    for (i=0; i<kn; ++i) BBT[i] = 0.;
  }

  if (*TRANSA == trans[0]) {
    BLASgemm_(&trans[0],&leaveB,M,K,N,ALPHA,CB,LDC,B,LDB,&one,AB,LDA);
  } else {
    BLASgemm_(&trans[0],&leaveB,M,K,N,ALPHA,CB,LDC,B,LDB,&one,ABT,LDA);
    for (i=0; i<*m; i++) {for (j=0; j<*k; j++) {AB[j+i*(*m)] = ABT[i+j*(*k)];}}
  }
  if (*TRANSB == trans[0]) {
    BLASgemm_(&leaveA,&trans[0],M,N,K,ALPHA,A,LDA,CB,LDC,&one,BB,LDB);
  } else {
    BLASgemm_(&leaveA,&trans[0],M,N,K,ALPHA,A,LDA,CB,LDC,&one,BBT,LDB);
    for (i=0; i<*k; i++) {for (j=0; j<*n; j++) {BB[j+i*(*m)] = BBT[i+j*(*k)];}}
  }
  ierr = PetscFree2(ABT,BBT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
