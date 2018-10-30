#include <petscmat.h>
#include "dgemm.c"

void printmat(const char *name,PetscInt m,PetscInt n,PetscScalar *A);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBLASInt   m=2,n=2,k=2;
  PetscScalar    alpha=1.,beta=0.;
  PetscScalar    *A,*B,*C,*Ad,*Bd,*Cd,*Ab,*Bb,*Cb;
  PetscInt       i,j;

  ierr = PetscInitialize(&argc,&argv,"petscoptions",NULL);CHKERRQ(ierr);
  ierr = PetscMalloc3(m*k,&A,k*n,&B,m*n,&C);CHKERRQ(ierr);
  ierr = PetscMalloc3(m*k,&Ad,k*n,&Bd,m*n,&Cd);CHKERRQ(ierr);
  ierr = PetscMalloc3(m*k,&Ab,k*n,&Bb,m*n,&Cb);CHKERRQ(ierr);

  printf("Initialise matrices A and B to be differentiated:\n");
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

  printf("\n\nInitialise forward seed matrices Ad and Bd:\n");
  for (i=0; i<m; ++i) {
    for (j=0; j<k; ++j) {
      Ad[i+j*k] = -i;
    }
  }
  printmat("Ad:",m,k,Ad);
  for (i=0; i<k; ++i) {
    for (j=0; j<n; ++j) {
      Bd[i+j*n] = 3*j;
    }
  }
  printmat("Bd:",k,n,Bd);

  printf("\n\nInitialise reverse seed matrix Cb:\n");
  for (i=0; i<m; ++i) {
    for (j=0; j<n; ++j) {
      Cb[i+j*n] = 2*i-j;
    }
  }
  printmat("Cb:",k,n,Cb);

  ierr = PetscDGEMMForward("N","N",&m,&n,&k,&alpha,A,Ad,&m,B,Bd,&k,&beta,C,Cd,&m);CHKERRQ(ierr);
  ierr = PetscDGEMMReverse("N","N",&m,&n,&k,&alpha,A,Ab,&m,B,Bb,&k,&beta,C,Cb,&m);CHKERRQ(ierr);

  printf("\n\nResults of multiplication and differentiation:\n");
  printmat("\nC = A*B:",m,n,C);
  printmat("\nCd:",m,n,Cd);
  printmat("\nAb:",m,n,Ab);
  printmat("\nBb:",m,n,Bb);

  ierr = PetscFree3(Ab,Bb,Cb);CHKERRQ(ierr);
  ierr = PetscFree3(Ad,Bd,Cd);CHKERRQ(ierr);
  ierr = PetscFree3(A,B,C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

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

