#ifndef EX16ADJ_LOADED
#define EX16ADJ_LOADED

void rhs(double f[2],const double *x,const double mu);
void ComputeJacobian(double f[],const double *x,const double mu,double J[2][2]);
void ComputeJacobianP(double f[],const double *x,const double mu,double J[2][1]);

#endif
