#include "ex16adj.h"

void rhs(double f[2],const double *x,const double mu)
{
  f[0] = x[1];
  f[1] = mu*(1.-x[0]*x[0])*x[1]-x[0];
}

