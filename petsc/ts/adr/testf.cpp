#include <petscts.h>
#include <iostream>

typedef struct {
  PetscScalar u,v;
} Field;

PetscErrorCode Shift(PetscScalar *arr,PetscInt m,PetscInt n,PetscInt mstart,PetscInt nstart,PetscScalar **a[])
{
  PetscErrorCode ierr;
  PetscInt       j;

  PetscFunctionBegin;

  for (j=0; j<m; j++) (*a)[j] = arr + j*n - nstart;
  (*a)[0] -= mstart;

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       s=0,gs=-1,m=5,gm=7,dof=2,i,j,k=0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;

  PetscScalar *arr = new PetscScalar[dof*gm*gm];        // Contiguous array of scalars
  Field  **A       = new Field*[gm];            	// Field containing same number of scalars

  for (j=0; j<gm; j++) {
    A[j] = new Field[gm];
    delete[] A[j];
  }

  ierr = Shift(arr,gm,gm*dof,gs,gs*dof,(PetscScalar***)A);CHKERRQ(ierr);

  for (j=gs; j<gm; j++) {
    for (i=gs; i<gm; i++) {
      std::cout << i << "," << j << " " << &A[j][i].u << " " << &arr[k++] << std::endl;
      std::cout << i << "," << j << " " << &A[j][i].v << " " << &arr[k++] << std::endl;
    }
  }

  delete[] A;
  delete[] arr;

  ierr = PetscFinalize();CHKERRQ(ierr);

  return ierr;
}

