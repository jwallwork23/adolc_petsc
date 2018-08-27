#ifndef EX5_LOADED
#define EX5_LOADED

typedef struct {
  double u,v;
} Field;

typedef struct {
  double D1,D2,gamma,kappa;
} AppCtx;

void RHSLocal(Field **f,Field **u,int xs,int xm,int ys,int ym,double hx,double hy,void *ptr);

#endif
