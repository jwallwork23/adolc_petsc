#include "contexts.cpp"
#include "sparse.cpp"
#include "init.cpp"


/* --------------------------------------------------------------------------------
   Drivers for RHSJacobian and IJacobian
   ----------------------------------------------------------------------------- */

/*
  Compute Jacobian for explicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is not used, full Jacobian is
  assembled (not recommended!).

  TODO: Make tape tag selectable

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
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);
  if (adctx->Seed)
    fov_forward(1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(1,m,n,u_vec,J);
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
          ierr = MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Compute Jacobian for explicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is not used, full Jacobian is
  assembled (not recommended!).

  TODO: Make tape tag selectable

  Input parameters:
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode AdolcComputeRHSJacobianLocal(Mat A,PetscScalar *u_vec,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);
  if (adctx->Seed)
    fov_forward(1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(1,m,n,u_vec,J);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian:",m,p,J);CHKERRQ(ierr);
      adctx->sparse_view_done = PETSC_TRUE;
    }
    ierr = RecoverJacobianLocal(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Compute Jacobian for implicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is not used, full Jacobian is
  assembled (not recommended!).

  TODO: Make tape tags selectable

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
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(1,m,n,u_vec,J);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian dF/dx:",m,p,J);CHKERRQ(ierr);
    }
    ierr = RecoverJacobian(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        if (fabs(J[i][j]) > 1.e-16) {
          ierr = MatSetValues(A,1,&i,1,&j,&J[i][j],INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* a * dF/d(xdot) part */
  if (adctx->Seed)
    fov_forward(2,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(2,m,n,u_vec,J);
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
          ierr = MatSetValues(A,1,&i,1,&j,&J[i][j],ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Compute local portion of Jacobian for implicit TS in compressed format and recover from this, using
  precomputed seed and recovery matrices. If sparse mode is not used, full Jacobian is
  assembled (not recommended!).

  TODO: Make tape tags selectable

  Input parameters:
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode AdolcComputeIJacobianLocal(Mat A,PetscScalar *u_vec,PetscReal a,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(1,m,n,u_vec,J);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian dF/dx:",m,p,J);CHKERRQ(ierr);
    }
    ierr = RecoverJacobianLocal(A,INSERT_VALUES,m,p,adctx->Rec,J,NULL);CHKERRQ(ierr);
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
  if (adctx->Seed)
    fov_forward(2,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(2,m,n,u_vec,J);
  if (adctx->sparse) {
    if ((adctx->sparse_view) && (!adctx->sparse_view_done)) {
      ierr = PrintMat(MPI_COMM_WORLD,"Compressed Jacobian dF/d(xdot):",m,p,J);CHKERRQ(ierr);
      adctx->sparse_view_done = PETSC_TRUE;
    }
    ierr = RecoverJacobianLocal(A,ADD_VALUES,m,p,adctx->Rec,J,&a);CHKERRQ(ierr);
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
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------
   Drivers for Jacobian w.r.t. a parameter
   ----------------------------------------------------------------------------- */

/*
  Compute Jacobian w.r.t a parameter for explicit TS.

  TODO: Do not form whole matrix. Just propagate [0,0,1] (for example).
  TODO: Account for multiple parameters
  TODO: Allow compressed format

  Input parameters:
  u_vec - vector at which to evaluate Jacobian
  param - the parameter
  tag   - tape identifier
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode AdolcComputeRHSJacobianP(Mat A,PetscScalar *u_vec,PetscScalar *param,PetscInt tag,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j = 0,m = adctx->m,n = adctx->n;
  PetscScalar    **J,*concat;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,n+1,&J);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+1,&concat);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    concat[i] = u_vec[i];
  }
  concat[n] = param[0];
  jacobian(tag,m,n+1,concat,J);
  for (i=0; i<m; i++) {
    //if (fabs(J[i][n]) > 1.e-16) {
      ierr = MatSetValues(A,1,&i,1,&j,&J[i][n],INSERT_VALUES);CHKERRQ(ierr);
    //}
  }
  ierr = PetscFree(concat);CHKERRQ(ierr);
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  Compute local portion of Jacobian w.r.t a parameter for explicit TS.

  TODO: Do not form whole matrix. Just propagate [0,0,1] (for example).
  TODO: Account for multiple parameters
  TODO: Allow compressed format

  Input parameters:
  u_vec - vector at which to evaluate Jacobian
  param - the parameter
  tag   - tape identifier
  ctx   - ADOL-C context, as defined above

  Output parameter:
  A     - Mat object corresponding to Jacobian
*/
PetscErrorCode AdolcComputeRHSJacobianPLocal(Mat A,PetscScalar *u_vec,PetscScalar *param,PetscInt tag,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j = 0,m = adctx->m,n = adctx->n;
  PetscScalar    **J,*concat;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,1,&J);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+1,&concat);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    concat[i] = u_vec[i];
  }
  concat[n] = param[0];
  jacobian(tag,m,n+1,concat,J);
  for (i=0; i<m; i++) {
    if (fabs(J[i][n]) > 1.e-16) {
      ierr = MatSetValuesLocal(A,1,&i,1,&j,&J[i][n],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(concat);CHKERRQ(ierr);
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* --------------------------------------------------------------------------------
   Drivers for Jacobian diagonal
   ----------------------------------------------------------------------------- */

/*
  Compute local portion of Jacobian diagonal for implicit TS in compressed format and recover
  from this, using precomputed seed matrix and recovery vector.

  TODO: Make tape tags selectable

  Input parameters:
  u_vec - vector at which to evaluate Jacobian
  ctx   - ADOL-C context, as defined above

  Output parameter:
  diag  - Vec object corresponding to Jacobian diagonal
*/
PetscErrorCode AdolcComputeIJacobianAndDiagonalLocal(Vec diag,PetscScalar *u_vec,PetscReal a,void *ctx)
{
  AdolcCtx       *adctx = (AdolcCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,m = adctx->m,n = adctx->n,p = adctx->p;
  PetscScalar    **J;

  PetscFunctionBegin;
  ierr = AdolcMalloc2(m,p,&J);CHKERRQ(ierr);

  /* dF/dx part */
  if (adctx->Seed)
    fov_forward(1,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(1,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverDiagonalLocal(diag,INSERT_VALUES,m,adctx->rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      if (fabs(J[i][i]) > 1.e-16) {
        ierr = VecSetValuesLocal(diag,1,&i,&J[i][i],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(diag);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(diag);CHKERRQ(ierr);

  /* a * dF/d(xdot) part */
  if (adctx->Seed)
    fov_forward(2,m,n,p,u_vec,adctx->Seed,NULL,J);
  else
    jacobian(2,m,n,u_vec,J);
  if (adctx->sparse) {
    ierr = RecoverDiagonalLocal(diag,ADD_VALUES,m,adctx->rec,J,NULL);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      if (fabs(J[i][i]) > 1.e-16) {
        J[i][i] *= a;
        ierr = VecSetValuesLocal(diag,1,&i,&J[i][i],ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecAssemblyBegin(diag);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(diag);CHKERRQ(ierr);
  ierr = AdolcFree2(J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

