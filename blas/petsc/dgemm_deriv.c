#include <petscblaslapack.h>


/*
  Forward derivative of BLASgemm w.r.t. both matrix inputs.

  The derivative code was initially obtained by transformation of the LAPACK Fortran source code
  using the Tapenade automatic differentiation tool. This was then re-interpreted in terms of calls
  to BLASgemm itself, so that no new BLAS level functions need be written.
*/
PetscErrorCode GEMMDot(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AD,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BD,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CD,PetscBLASInt *LDC)
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
  Forward derivative of BLASgemm w.r.t. first matrix input, A.
*/
PetscErrorCode GEMMDotA(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AD,PetscBLASInt *LDA,PetscScalar *B,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CD,PetscBLASInt *LDC)
{
  PetscScalar    zero = 0.;
  PetscInt       *m = (PetscInt*) M,*n = (PetscInt*) N,i;

  PetscFunctionBegin;

  /* Undifferentiated call */
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

  /* Differentiated call */
  for (i=0; i<(*m)*(*n); ++i) CD[i] = 0.;
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,AD,LDA,B,LDB,&zero,CD,LDC);
  PetscFunctionReturn(0);
}

/*
  Forward derivative of BLASgemm w.r.t. second matrix input, B.
*/
PetscErrorCode GEMMDotB(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BD,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CD,PetscBLASInt *LDC)
{
  PetscScalar    zero = 0.;
  PetscInt       *m = (PetscInt*) M,*n = (PetscInt*) N,i;

  PetscFunctionBegin;

  /* Undifferentiated call */
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);

  /* Differentiated call */
  for (i=0; i<(*m)*(*n); ++i) CD[i] = 0.;
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,BD,LDB,&zero,CD,LDC);
  PetscFunctionReturn(0);
}

/*
  Reverse derivative of BLASgemm w.r.t. both matrix inputs.

  The derivative code was initially obtained by transformation of the LAPACK Fortran source code
  using the Tapenade automatic differentiation tool. This was then re-interpreted in terms of calls
  to BLASgemm itself, so that no new BLAS level functions need be written.
*/
PetscErrorCode GEMMBar(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AB,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BB,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CB,PetscBLASInt *LDC)
{
  PetscScalar    zero = 0.;
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
    BLASgemm_(&trans[0],&leaveB,M,K,N,ALPHA,CB,LDC,B,LDB,&zero,AB,LDA);
  } else {
    leaveA = trans[0];
    BLASgemm_(TRANSB,&trans[1],K,M,N,ALPHA,B,LDB,CB,LDC,&zero,AB,LDA);
  }

  /* Reverse derivative for B */
  if (*TRANSB == trans[0])
    BLASgemm_(&leaveA,&trans[0],M,N,K,ALPHA,A,LDA,CB,LDC,&zero,BB,LDB);
  else
    BLASgemm_(&trans[1],TRANSA,N,M,K,ALPHA,CB,LDC,A,LDA,&zero,BB,LDB);

  /* Quick return scaled C */
  BLASgemm_(&trans[0],&trans[0],M,N,K,&zero,A,LDA,B,LDB,BETA,CB,LDC);
  PetscFunctionReturn(0);
}

/*
  Reverse derivative of BLASgemm w.r.t. first matrix input, A.
*/
PetscErrorCode GEMMBarA(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscScalar *AB,PetscBLASInt *LDA,PetscScalar *B,,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CB,PetscBLASInt *LDC)
{
  PetscScalar    zero = 0.;
  PetscInt       *m = (PetscInt*) M,*n = (PetscInt*) N,*k = (PetscInt*) K,i;
  char           leaveB,trans[2] = "NT";

  PetscFunctionBegin;
  if (*TRANSB == trans[0])
    leaveB = trans[1];
  else
    leaveB = trans[0];
  for (i=0; i<(*m)*(*k); ++i) AB[i] = 0.;

  /* Reverse derivative for A */
  if (*TRANSA == trans[0]) {
    leaveA = trans[1];
    BLASgemm_(&trans[0],&leaveB,M,K,N,ALPHA,CB,LDC,B,LDB,&zero,AB,LDA);
  } else {
    leaveA = trans[0];
    BLASgemm_(TRANSB,&trans[1],K,M,N,ALPHA,B,LDB,CB,LDC,&zero,AB,LDA);
  }

  /* Quick return scaled C */
  BLASgemm_(&trans[0],&trans[0],M,N,K,&zero,A,LDA,B,LDB,BETA,CB,LDC);
  PetscFunctionReturn(0);
}

/*
  Reverse derivative of BLASgemm w.r.t. second matrix input, B.
*/
PetscErrorCode GEMMBarB(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscBLASInt *LDA,PetscScalar *B,PetscScalar *BB,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscScalar *CB,PetscBLASInt *LDC)
{
  PetscScalar    zero = 0.;
  PetscInt       *m = (PetscInt*) M,*n = (PetscInt*) N,*k = (PetscInt*) K,i;
  char           leaveA,trans[2] = "NT";

  PetscFunctionBegin;
  if (*TRANSA == trans[0])
    leaveA = trans[1];
  else
    leaveA = trans[0];
  for (i=0; i<(*k)*(*n); ++i) BB[i] = 0.;

  /* Reverse derivative for B */
  if (*TRANSB == trans[0])
    BLASgemm_(&leaveA,&trans[0],M,N,K,ALPHA,A,LDA,CB,LDC,&zero,BB,LDB);
  else
    BLASgemm_(&trans[1],TRANSA,N,M,K,ALPHA,CB,LDC,A,LDA,&zero,BB,LDB);

  /* Quick return scaled C */
  BLASgemm_(&trans[0],&trans[0],M,N,K,&zero,A,LDA,B,LDB,BETA,CB,LDC);
  PetscFunctionReturn(0);
}

/*
  Wrapper for pair of BLASgemm calls, representing matrix-tensor-matrix-vector product
    vec(V) := alpha * (A \otimes B) * vec(U) + beta * vec(V)  <=> V = alpha * B*U*A^T + beta * V

  NOTES:
  - Block dimensions are assumed square and identical.
  - Memory for the work array tmp should be preallocated.
*/
PetscErrorCode MTMV(const PetscBLASInt M,PetscScalar alpha,PetscScalar **A,PetscScalar **B,PetscScalar **U,PetscScalar beta,PetscScalar **tmp,PetscScalar **V)
{
  PetscScalar one = 1.,zero = 0.;

  PetscFunctionBegin;
  BLASgemm_("N","N",&M,&M,&M,&one,&B[0][0],&M,&U[0][0],&M,&zero,&tmp[0][0],&M);
  BLASgemm_("N","T",&M,&M,&M,&alpha,&tmp[0][0],&M,&A[0][0],&M,&beta,&V[0][0],&M);
  PetscFunctionReturn(0);
}

/*
  Forward mode derivative of MTMV w.r.t. the matrix argument U, given seed matrix Udot:
    vec(Vdot) = alpha * (A \otimes B) vec(Udot)  <=> Vdot = alpha * (B^T * Udot * A)
*/
PetscErrorCode MTMVDot(const PetscBLASInt M,PetscScalar alpha,PetscScalar **A,PetscScalar **B,PetscScalar **U,PetscScalar **Udot,PetscScalar beta,PetscScalar **tmp,PetscScalar **V,PetscScalar **Vdot)
{
  PetscScalar one = 1.,zero = 0.;

  PetscFunctionBegin;

  /* Undifferentiated code */
  if ((U) && (V)) {
    BLASgemm_("N","N",&M,&M,&M,&one,&B[0][0],&M,&U[0][0],&M,&zero,&tmp[0][0],&M);
    BLASgemm_("N","T",&M,&M,&M,&alpha,&tmp[0][0],&M,&A[0][0],&M,&beta,&V[0][0],&M);
  }

  /* Differentiated code */
  BLASgemm_("N","N",&M,&M,&M,&one,&B[0][0],&M,&Udot[0][0],&M,&zero,&tmp[0][0],&M);
  BLASgemm_("N","T",&M,&M,&M,&alpha,&tmp[0][0],&M,&A[0][0],&M,&beta,&Vdot[0][0],&M);
  PetscFunctionReturn(0);
}

/*
  Reverse mode derivative of MTMV w.r.t. the matrix argument U, given a seed matrix Vdot:
    vec(Ubar) = alpha * (A^T \otimes B^T) vec(Vbar)  <=>  alpha * (A^T * Vbar * B)

  NOTE the transposition of A and B: ["N","N";"N","T"] -> ["T","N";"N","N"]
*/
PetscErrorCode MTMVBar(const PetscBLASInt M,PetscScalar alpha,PetscScalar **A,PetscScalar **B,PetscScalar **U,PetscScalar **Ubar,PetscScalar beta,PetscScalar **tmp,PetscScalar **V,PetscScalar **Vbar)
{
  PetscScalar one = 1.,zero = 0.;

  PetscFunctionBegin;
  BLASgemm_("T","N",&M,&M,&M,&one,&B[0][0],&M,&Ubar[0][0],&M,&zero,&tmp[0][0],&M);
  BLASgemm_("N","N",&M,&M,&M,&alpha,&tmp[0][0],&M,&A[0][0],&M,&beta,&Vbar[0][0],&M);
  PetscFunctionReturn(0);
}
