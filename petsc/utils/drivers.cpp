#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>
#include "sparse.cpp"

#define tag 1


typedef struct {
  PetscBool   zos,zos_view,no_an,sparse,sparse_view,sparse_view_done;
  PetscScalar **Seed,**Rec;
  PetscInt    p;
} AdolcCtx;


/*@C
  Default method of computing full Jacobian without exploiting sparsity (not recommended!)

  Input parameters:
  m,n   - number of dependent and independent variables, respectively
  u_vec - vector at which to evaluate Jacobian

  Output parameter:
  A     - Mat object corresponding to Jacobian
@*/
PetscErrorCode AdolcComputeJacobian(Mat A,PetscInt m,PetscInt n,PetscScalar *u_vec)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscScalar    **J;

  PetscFunctionBegin;
  J = myalloc2(m,n);
  jacobian(tag,m,n,u_vec,J);
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      if (fabs(J[i][j]) > 1.e-16) {
        ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  myfree2(J);
  PetscFunctionReturn(0);
}

/*@C
  Compute Jacobian in compressed format and recover from this, using precomputed seed and
  recovery matrices.

  Input parameters:
  m,n   - number of dependent and independent variables, respectively
  u_vec - vector at which to evaluate Jacobian

  Output parameter:
  A     - Mat object corresponding to Jacobian
@*/
PetscErrorCode AdolcComputeJacobian(Mat A,PetscInt m,PetscInt n,PetscScalar *u_vec,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscScalar    **J,*f_vec;

  PetscFunctionBegin;
  ierr = PetscMalloc1(m,&f_vec);CHKERRQ(ierr);
  J = myalloc2(m,adctx->p);
  fov_forward(tag,m,n,adctx->p,u_vec,adctx->Seed,f_vec,J);
  ierr = PetscFree(f_vec);CHKERRQ(ierr);
  if (adctx->sparse_view) {
    if (!adctx->sparse_view_done) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian:",m,adctx->p,J);CHKERRQ(ierr);
      adctx->sparse_view_done = PETSC_TRUE;
    }
  }
  ierr = RecoverJacobian(A,m,adctx->p,adctx->Rec,J);CHKERRQ(ierr);
  myfree2(J);
  PetscFunctionReturn(0);
}
