#include <petscmat.h>
#include "dgemm.c"

void printmat(const char *name,PetscInt m,PetscInt n,PetscScalar *A)
{
  int i,j;

  printf("%s\n",name);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      printf("%.1f,",A[i+j*n]);
    }
    printf("\n");
  }
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBLASInt   m=2,n=2,k=2;
  PetscScalar    alpha=1.,beta=0.;
  PetscScalar    *A,*B,*C;
  PetscInt       i,j;

  ierr = PetscInitialize(&argc,&argv,"petscoptions",NULL);CHKERRQ(ierr);
  ierr = PetscMalloc3(m*k,&A,k*n,&B,m*n,&C);CHKERRQ(ierr);

  for (i=0; i<m; ++i) {
    for (j=0; j<k; ++j) {
      A[i+j*k] = i+1;
    }
  }
  printmat("A:",m,k,A);
  for (i=0; i<k; ++i) {
    for (j=0; j<n; ++j) {
      B[i+j*n] = j+1;
    }
  }
  printmat("B:",k,n,B);
  ierr = PetscDGEMM("N","N",&m,&n,&k,&alpha,A,&m,B,&k,&beta,C,&m);CHKERRQ(ierr);
  printmat("C:",m,n,C);

  ierr = PetscFree3(A,B,C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
