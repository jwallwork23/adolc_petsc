#include "contexts.cpp"
#include "sparse.cpp"

#define tag 1

/*@C
  Compute Jacobian in compressed format and recover from this, using precomputed seed and
  recovery matrices. If sparse mode is used, full Jacobian is assembled (not recommended!).

  Input parameters:
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
@*/
PetscErrorCode AdolcComputeRHSJacobian(Mat A,PetscScalar *u_vec,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;

  J = myalloc2(m,p);
  fov_forward(tag,m,n,p,u_vec,adctx->Seed,NULL,J);
  //jacobian(tag,m,n,u_vec,J);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian:",m,p,J);CHKERRQ(ierr);
      adctx->sparse_view_done = PETSC_TRUE;
    }
    ierr = RecoverJacobian(A,m,p,adctx->Rec,J);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  myfree2(J);
  PetscFunctionReturn(0);
}

// TODO: IJacobian, using dependence upon udot
