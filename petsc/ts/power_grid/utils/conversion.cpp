#include <petscdm.h>

/* Converts from machine frame (dq) to network (phase a real,imag) reference frame */
template <class T>
PetscErrorCode dq2ri(T Fd,T Fq,T delta,T *Fr,T *Fi)
{
  PetscFunctionBegin;
  *Fr =  Fd*PetscSinScalar(delta) + Fq*PetscCosScalar(delta);
  *Fi = -Fd*PetscCosScalar(delta) + Fq*PetscSinScalar(delta);
  PetscFunctionReturn(0);
}

/* Converts from network frame ([phase a real,imag) to machine (dq) reference frame */
template <class T>
PetscErrorCode ri2dq(T Fr,T Fi,T delta,T *Fd,T *Fq)
{
  PetscFunctionBegin;
  *Fd =  Fr*PetscSinScalar(delta) - Fi*PetscCosScalar(delta);
  *Fq =  Fr*PetscCosScalar(delta) + Fi*PetscSinScalar(delta);
  PetscFunctionReturn(0);
}

