#include "speelpenning.h"

void product(double *y, double x[], int size){
  int i;
  for(i=0; i<size; i++)
    *y *= x[i];
}
