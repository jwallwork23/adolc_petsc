#include <adolc/adolc.h>	/* ##### Include ADOL-C ##### */
#include <stdio.h>

void RHSFunction(double f[2],const double x[3])
{
  adouble f_a[2];	// ##### adouble for dependent variables #####
  adouble x_a[3];   	// ##### adouble for independent variables #####

  trace_on(1);
  x_a[0] <<= x[0]; x_a[1] <<= x[1]; x_a[2] <<= x[2];
  f_a[0] = x_a[1];
  f_a[1] = x_a[2]*(1.-x_a[0]*x_a[0])*x_a[1]-x_a[0];
  f_a[0] >>= f[0]; f_a[1] >>= f[1];
  trace_off();

}

int main()
{

  // initialisation
  int i;
  int m = 2;	// number of dependent variables
  int n = 3;	// number of independent variables
  double f[2];
  const double x0[3] = {2., 1., 1.};
  const double *indep = x0;

  // function evaluation
  RHSFunction(f,x0);
  printf("x = [%.4f, %.4f], mu = %.4f\n",x0[0],x0[1],x0[2]);
  printf("x = [%.4f",*(indep++));
  printf(", %.4f], ",*(indep++));
  printf("mu = %.4f\n",*indep);
  printf("f(x,mu) = [%.4f, %.4f]\n",f[0],f[1]);

  // Jacobian calculation
  double** Jx = (double**) malloc(m*sizeof(double*));
  for(i=0;i<m;i++)
    Jx[i] = (double*)malloc(n*sizeof(double));
  jacobian(1,m,n,x0,Jx);
  
  // analytic Jacobian
  double J[m][n];
  J[0][0] = 0;
  J[0][1] = 1.;
  J[0][2] = 0.;
  J[1][0] = -2.*x0[2]*x0[1]*x0[0]-1.;
  J[1][1] = x0[2]*(1.0-x0[0]*x0[0]);
  J[1][2] = (1.-x0[0]*x0[0])*x0[1];

  // compare Jacobians
  printf("J_{exact} =\n    [%.4f, %.4f, %.4f]\n    [%.4f, %.4f, %.4f]\n",J[0][0],J[0][1],J[0][2],J[1][0],J[1][1],J[1][2]);
  printf("J_{adolc} =\n    [%.4f, %.4f, %.4f]\n    [%.4f, %.4f, %.4f]\n\n",Jx[0][0],Jx[0][1],Jx[0][2],Jx[1][0],Jx[1][1],Jx[1][2]);

  return 0;
}
