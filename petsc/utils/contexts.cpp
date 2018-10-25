#include <petscdm.h>
#include <petscdmda.h>
#include <adolc/adolc.h>


#ifndef ADOLCCTX
#define ADOLCCTX
typedef struct {
  /* Zero Order Scalar (ZOS) test */
  PetscBool   zos,zos_view;

  /* No ADOL-C annotation */
  PetscBool   no_an;

  /* Compressed Jacobian computation */
  PetscBool   sparse,sparse_view,sparse_view_done;
  PetscScalar **Seed,**Rec,*rec;
  PetscInt    p;

  /* Matrix dimensions */
  PetscInt    m,n;
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
  PetscBool flg;
} MatCtx;
#endif
