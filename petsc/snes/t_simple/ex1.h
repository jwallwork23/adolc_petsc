#ifndef EX1_LOADED
#define EX1_LOADED

void f1(double ff[2],const double *xx);
void f2(double ff[2],const double *xx);
void ComputeJacobian1(double ff[2],const double *xx,double J[4]);
void ComputeJacobian2(double ff[2],const double *xx,double J[4]);
// void ComputeJacobian(double ff[2],const double *xx,double J[2][2]);

#endif
