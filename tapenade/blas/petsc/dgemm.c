#include <petscblaslapack.h>

PetscErrorCode PetscDGEMM(const char* TRANSA,const char* TRANSB,PetscBLASInt *M,PetscBLASInt *N,PetscBLASInt *K,PetscScalar *ALPHA,PetscScalar *A,PetscBLASInt *LDA,PetscScalar *B,PetscBLASInt *LDB,PetscScalar *BETA,PetscScalar *C,PetscBLASInt *LDC)
{
  PetscFunctionBegin;
  BLASgemm_(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC);
  PetscFunctionReturn(0);
}

