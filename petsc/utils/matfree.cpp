#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include <adolc/adolc.h>

/*@C
  Jacobian transpose vector product, provided matrix free using reverse mode of AD

  Input parameters:
  J - MatShell type Jacobian matrix
  x - vector to be multiplied
  y - Jacobian transpose vector product
@*/
PetscErrorCode JacobianTransposeVectorProduct(Mat J,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PetscInt          i,m,n,q = 1;
  const PetscScalar *data;
  PetscScalar       **datarray,**action;

  PetscFunctionBegin;

  /* Read data and allocate memory */
  ierr = MatGetSize(J,&m,&n);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&data);CHKERRQ(ierr);
  ierr = AdolcMalloc2(q,m,&datarray);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&data);CHKERRQ(ierr);
  for (i=0; i<m; i++)
    datarray[0][i] = data[i];        // TODO: Create a wrapper for this
  ierr = VecRestoreArrayRead(x,&data);CHKERRQ(ierr);

  /* Compute action and set values in vector*/
  ierr = AdolcMalloc2(q,n,&action);CHKERRQ(ierr);
  fov_reverse(tag,m,n,q,uarray,action);
  for (i=0; i<n; i++) {
    ierr = VecSetValues(y,1,&i,&action[0][i],INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Free memory */
  ierr = AdolcFree2(action);CHKERRQ(ierr);
  ierr = AdolcFree2(datarray);CHKERRQ(ierr);
  ierr = PetscFree(data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

