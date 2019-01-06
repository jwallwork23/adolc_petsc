#include "tracing.cxx"


// TODO: Consider matrix-free B for preconditioner?
PetscErrorCode IJacobianMatFree(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A_shell,Mat B,void *ctx)
{
  MatCtx            *mctx;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A_shell,(void **)&mctx);CHKERRQ(ierr);

  mctx->time  = t;
  mctx->shift = a;
  if (mctx->ts != ts) mctx->ts = ts;
  ierr = VecCopy(X,mctx->X);CHKERRQ(ierr);
  ierr = VecCopy(Xdot,mctx->Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
