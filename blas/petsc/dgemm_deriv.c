#include <petscblaslapack.h>


/*
  Forward derivative of BLASgemm w.r.t. both A and B.

  The derivative code was initially obtained by transformation of the LAPACK Fortran source code
  using the Tapenade automatic differentiation tool. This was then re-interpreted in terms of calls
  to BLASgemm itself, so that no new BLAS level functions need be written.
*/
PetscErrorCode PetscGEMMForward(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AD,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BD,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CD,PetscBLASInt *LDC)
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

/*
  Forward derivative of BLASgemm w.r.t. first input, A.
*/
PetscErrorCode PetscGEMMForward1(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AD,PetscBLASInt *LDA,PetscScalar *B,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CD,PetscBLASInt *LDC)
{
  PetscScalar    one = 1.;
  PetscInt       *m = (PetscInt*) M,*n = (PetscInt*) N,i;

  PetscFunctionBegin;

  /* Undifferentiated call */
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

  /* Differentiated call */
  for (i=0; i<(*m)*(*n); ++i) CD[i] = 0.;
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,AD,LDA,B,LDB,BETA,CD,LDC);
  PetscFunctionReturn(0);
}

/*
  Forward derivative of BLASgemm w.r.t. second input, B.
*/
PetscErrorCode PetscGEMMForward2(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BD,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CD,PetscBLASInt *LDC)
{
  PetscScalar    one = 1.;
  PetscInt       *m = (PetscInt*) M,*n = (PetscInt*) N,i;

  PetscFunctionBegin;

  /* Undifferentiated call */
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

  /* Differentiated call */
  for (i=0; i<(*m)*(*n); ++i) CD[i] = 0.;
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,BD,LDB,&BETA,CD,LDC);
  PetscFunctionReturn(0);
}

/*
  Reverse derivative of BLASgemm.

  The derivative code was initially obtained by transformation of the LAPACK Fortran source code
  using the Tapenade automatic differentiation tool. This was then re-interpreted in terms of calls
  to BLASgemm itself, so that no new BLAS level functions need be written.
*/
PetscErrorCode PetscGEMMReverse(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AB,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BB,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CB,PetscBLASInt *LDC)
{
  PetscScalar    one = 1.;
  PetscInt       *m = (PetscInt*) M,*n = (PetscInt*) N,*k = (PetscInt*) K,i;
  char           leaveA,leaveB,trans[2] = "NT";

  PetscFunctionBegin;
  if (*TRANSB == trans[0])
    leaveB = trans[1];
  else
    leaveB = trans[0];
  for (i=0; i<(*m)*(*k); ++i) AB[i] = 0.;
  for (i=0; i<(*k)*(*n); ++i) BB[i] = 0.;

  /* Reverse derivative for A */
  if (*TRANSA == trans[0]) {
    leaveA = trans[1];
    BLASgemm_(&trans[0],&leaveB,M,K,N,ALPHA,CB,LDC,B,LDB,&one,AB,LDA);
  } else {
    leaveA = trans[0];
    BLASgemm_(TRANSB,&trans[1],K,M,N,ALPHA,B,LDB,CB,LDC,&one,AB,LDA);
  }

  /* Reverse derivative for B */
  if (*TRANSB == trans[0])
    BLASgemm_(&leaveA,&trans[0],M,N,K,ALPHA,A,LDA,CB,LDC,&one,BB,LDB);
  else
    BLASgemm_(&trans[1],TRANSA,N,M,K,ALPHA,CB,LDC,A,LDA,&one,BB,LDB);
  PetscFunctionReturn(0);
}
