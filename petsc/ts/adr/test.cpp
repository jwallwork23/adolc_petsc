#include <petscts.h>
#include <adolc/adolc.h>

typedef struct {
  adouble u,v;
} aField;


PetscErrorCode Shift(adouble *arr,PetscInt s,PetscInt m,PetscInt dof,adouble **a[])
{
  PetscErrorCode ierr;
  PetscInt       j;

  PetscFunctionBegin;

  for (j=0; j<m; j++) (*a)[j] = arr + dof*j*m - dof*s;
  *a -= s;

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       s=0,gs=-1,m=5,gm=7,dof=2,i,j,k=0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;

  adouble *arr = new adouble[dof*gm*gm];	// Contiguous array of adoubles
  aField **A = new aField*[gm];			// aField containing same number of adoubles

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

  ierr = PetscFinalize();CHKERRQ(ierr);

  return ierr;
}
