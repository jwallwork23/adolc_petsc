#ifndef EX5_LOADED
#define EX5_LOADED

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  PetscReal D1,D2,gamma,kappa;
} AppCtx;

void rhs(Field **f,const Field **x,int xs,int xm,int ys,int ym,void*);

#endif
