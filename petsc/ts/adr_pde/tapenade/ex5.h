#ifndef EX5_LOADED
#define EX5_LOADED

typedef struct {
  double u,v;
} Field;

typedef struct {
  double D1,D2,gamma,kappa;
} AppCtx;

void RHSLocal(Field **f,Field **u,int gxs,int gxm,int gys,int gym,double hx,double hy,void *ptr);
void ComputeJacobian(Field **f,Field **u,int gxs,int gxm,int gys,int gym,double hx,double hy,int My,void *ptr,double **J);

#endif
