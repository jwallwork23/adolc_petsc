#include <iostream>
#include <math.h>
using namespace std;

#define ADOLC_TAPELESS
#define NUMBER_DIRECTIONS 5
#include <adolc/adtl.h>
using namespace adtl;

int main()
{
  int n,i,j;

  cout << "SPEELPENNINGS PRODUCT (ADOL-C Documented Example)\n\n";
  cout << "number of independent variables = ? ";
  cin >> n;
  cout << endl;
  adtl::setNumDir(n);			    // FIXME: This doesn't work

  double  *xp = new double[n];
  double  yp = 0.0;
  adouble *x = new adouble[n];
  double  *e = new double[NUMBER_DIRECTIONS];
  adouble y = 1;

  cout << "x = [";
  for(i=0; i<n; i++) {
    xp[i] = (i+1.0)/(2.0+i);
    cout << xp[i] << ", ";
  }
  cout << "]" << endl;

  for(i=0; i<n; i++) {
    x[i].setValue(xp[i]);
    for (j=0; j<n; j++) {if (i==j) {e[j] = 1.;} else {e[j] = 0.;}}
    x[i].setADValue(e);
  }

  for (i=0; i<n; i++)
    y *= x[i];

  yp = y.getValue();
  cout << "y = " << yp << endl;
  double *g = (double*) y.getADValue();

  double y_d = 0;
  for(i=0; i<n; i++)
    y_d += g[i] * xp[i];             // y' evaluated at x
  cout << "y'(x) = " << y_d << "\n"; // This can also be computed easily using scalar tapeless mode

  double errg = 0;
  cout << "gradient = [";
  for (i=0; i<n; i++) {
    cout << g[i] << ", ";
    errg += fabs(g[i]-yp/xp[i]);     // vanishes analytically.
  }
  cout << "]" << endl;

  cout << yp-1/(1.0+n) << " error in function \n";
  cout << errg <<" error in gradient \n";


  delete[] e;
  delete[] x;
  delete[] xp;

  return 0;
}
