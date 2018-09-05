#include <petscts.h>
#include <adolc/adolc.h>

typedef struct {
  adouble u,v;
} aField;


PetscErrorCode Shift(adouble *arr,PetscInt s,PetscInt m,PetscInt dof,adouble **a[])
{
  PetscInt       j;

  PetscFunctionBegin;

  for (j=0; j<m; j++) (*a)[j] = arr + dof*j*m - dof*s;
  (*a)[0] -= s;

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscScalar    w;
  PetscInt       m=5,G,L;
  Vec            V;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;

  w = 1.;

  ierr = VecCreate(PETSC_COMM_WORLD,&V);CHKERRQ(ierr);
  ierr = VecSetSizes(V,PETSC_DECIDE,m*m*2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(V);CHKERRQ(ierr);
  ierr = VecSetUp(V);CHKERRQ(ierr);
  ierr = VecSet(V,w);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(V);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(V);CHKERRQ(ierr);
  ierr = VecGetSize(V,&G);CHKERRQ(ierr);
  ierr = VecGetLocalSize(V,&L);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);

  std::cout << "Global size = " << G << std::endl;
  std::cout << "Local size = " << L << std::endl;

/*
  PetscInt       s=0,gs=-1,m=5,gm=7,dof=2,i,j,k=0,N;

  adouble *arr = new adouble[dof*gm*gm];	// Contiguous array of adoubles
  aField  **A = new aField*[gm];		// aField containing same number of adoubles

  for (j=0; j<gm; j++) {
    A[j] = new aField[gm];
    delete[] A[j];
  }

  ierr = Shift(arr,gs,gm,dof,(adouble***)A);CHKERRQ(ierr);

  for (j=gs; j<gm; j++) {
    for (i=gs; i<gm; i++) {
      std::cout << i << "," << j << " " << &A[j][i].u << " " << &arr[k++] << std::endl;
      std::cout << i << "," << j << " " << &A[j][i].v << " " << &arr[k++] << std::endl;
    }
  }

  delete[] A;
  delete[] arr;
*/
  ierr = PetscFinalize();CHKERRQ(ierr);

  return ierr;
}
