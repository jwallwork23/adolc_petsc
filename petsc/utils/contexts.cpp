#include <petscdm.h>
#include <petscdmda.h>
#include <adolc/adolc.h>

#ifndef ADOLCCTX
#define ADOLCCTX
typedef struct {
  PetscBool   zos,zos_view,no_an,sparse,sparse_view,sparse_view_done;
  PetscScalar **Seed,**Rec;
  PetscInt    m,n,p;
} AdolcCtx;
#endif

/* Matrix (free) context */
#ifndef MATCTX
#define MATCTX
typedef struct {
  PetscReal time;
  Vec       X;
  Vec       Xdot;
  PetscReal shift;
  PetscInt  m,n;
  TS        ts;
} MatCtx;
#endif
