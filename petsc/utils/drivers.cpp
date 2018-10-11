#include "contexts.cpp"
#include "sparse.cpp"

/*
  Compute Jacobian for explicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is used, full Jacobian is
  assembled (not recommended!).

  Input parameters:
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode AdolcComputeRHSJacobian(Mat A,PetscScalar *u_vec,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;

  J = myalloc2(m,p);
  fov_forward(1,m,n,p,u_vec,adctx->Seed,NULL,J);
  //jacobian(1,m,n,u_vec,J);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian:",m,p,J);CHKERRQ(ierr);
      adctx->sparse_view_done = PETSC_TRUE;
    }
    ierr = RecoverJacobian(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
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
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Compute Jacobian for implicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is used, full Jacobian is
  assembled (not recommended!).

  Input parameters:
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode AdolcComputeIJacobian(Mat A,PetscScalar *u_vec,PetscReal a,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  J = myalloc2(m,p);

  /* dF/dx part */
  fov_forward(1,m,n,p,u_vec,adctx->Seed,NULL,J);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian dF/dx:",m,p,J);CHKERRQ(ierr);
    }
    ierr = RecoverJacobian(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* a * dF/d(xdot) part */
  fov_forward(2,m,n,p,u_vec,adctx->Seed,NULL,J);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian dF/d(xdot):",m,p,J);CHKERRQ(ierr);
      adctx->sparse_view_done = PETSC_TRUE;
    }
    ierr = RecoverJacobian(A,ADD_VALUES,m,p,adctx->Rec,J,&a);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          J[i][j] *= a;
          ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  myfree2(J);
  PetscFunctionReturn(0);
}

/*
  Compute diagonal of Jacobian for implicit TS in compressed format and recover from this, using
  precomputed seed matrix and recovery vector.

  Input parameters:
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian // TODO: Use Vec. Do matfree
*/
PetscErrorCode AdolcComputeIJacobianDiagonal(Mat A,PetscScalar *u_vec,PetscReal a,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  J = myalloc2(m,p);

  /* dF/dx part */
  fov_forward(1,m,n,p,u_vec,adctx->Seed,NULL,J);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian dF/dx:",m,p,J);CHKERRQ(ierr);
    }
    ierr = RecoverDiagonal(A,INSERT_VALUES,m,adctx->rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      if (fabs(J[i][i]) > 1.e-16) {
        ierr = MatSetValuesLocal(A,1,&i,1,&i,&J[i][i],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* a * dF/d(xdot) part */
  fov_forward(2,m,n,p,u_vec,adctx->Seed,NULL,J);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian dF/d(xdot):",m,p,J);CHKERRQ(ierr);
      adctx->sparse_view_done = PETSC_TRUE;
    }
    ierr = RecoverDiagonal(A,ADD_VALUES,m,adctx->rec,J,&a);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      if (fabs(J[i][i]) > 1.e-16) {
        J[i][i] *= a;
        ierr = MatSetValuesLocal(A,1,&i,1,&i,&J[i][i],ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  myfree2(J);
  PetscFunctionReturn(0);
}
