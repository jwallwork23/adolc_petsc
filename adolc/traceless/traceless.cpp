#include <iostream>
#include <math.h>
using namespace std;

int main()
{
  double x[3],y[3];

  x[0] = 1.;
  x[1] = 2.;
  x[2] = 3.;

  y[0] = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  y[1] = atan(sqrt(x[0]*x[0]+x[1]*x[1])/x[2]);
  y[2] = atan(x[1]/x[0]);

  cout << "y1=" << y[0] << ", y2=" << y[1] << ", y3=" << y[2] << endl;

  return 0;
}
