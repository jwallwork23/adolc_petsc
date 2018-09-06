#include <petscts.h>
#include <adolc/adolc.h>

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

  ierr = PetscFinalize();CHKERRQ(ierr);

  return ierr;
}
