#include <iostream>
#include <math.h>
using namespace std;

#define ADOLC_TAPELESS
#include <adolc/adtl.h>
using namespace adtl;

int main()
{
  adouble x[3],y[3];

  double one = 1.;

  x[0] = 1.;
  x[1] = 2.;
  x[2] = 3.;

  x[0].setADValue(&one);

  y[0] = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  y[1] = atan(sqrt(x[0]*x[0]+x[1]*x[1])/x[2]);
  y[2] = atan(x[1]/x[0]);

  cout << "y = ";
  for (int i=0; i<3; i++) {
    cout << y[i].getValue() << ", ";
  }
  cout << endl;
  cout << "y'= ";
  for (int i=0; i<3; i++) {
    const double *out = y[i].getADValue();
    cout << *out << ", ";
  }
  cout << endl;

  return 0;
}
