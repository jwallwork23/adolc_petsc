#ifndef EX5_LOADED
#define EX5_LOADED

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  PetscReal D1,D2,gamma,kappa;
} AppCtx;

void RHSLocal(Field **f,Field **u,PetscInt xs,PetscInt xm,PetscInt ys,PetscInt ym,PetscReal hx,PetscReal hy,void *ptr);

#endif
