#include <math.h>
#include "ex1.h"

void f2(double ff[2],const double *xx)
{
  ff[0] = sin(3.0 * xx[0]) + xx[0];
  ff[1] = xx[1];
}

