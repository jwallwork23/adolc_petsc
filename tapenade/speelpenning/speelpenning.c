#include <stdio.h>
#include <math.h>
#include "speelpenning_d.h"

int main(){
  int n,i;

  printf("speelpennings product (adol-c documented example)\n\n");
  printf("number of independent variables =?  \n");
  scanf("%d",&n);

  double x[n], xd[n];
  double y = 1, yd;
  double g[n], err;

  for(i=0; i<n; i++)
    x[i] = (i+1.0)/(2.0+i);	// some initialisation
  
  printf("gradient = [");
  for(i=0; i<n; i++){
    xd[i] = 1;
    product_d(&y,&yd,x,xd,n);
    xd[i] = 0;
    g[i] = yd;
    printf("%f, ",g[i]);
    y = 1;			// resetting these variables
    yd = 0;			//   is very important!
  }
  printf("]\n");

  product_d(&y,&yd,x,x,n);	// y' evaluated at x (i.e. xd = x)
  printf("y'(x) = %e\n",yd);

  err = y-1/(1.0+n);
  printf("%e error in function\n",err);
  err = 0;
  for(i=0; i<n; i++)
    err += fabs(g[i]-y/x[i]);
  printf("%e error in gradient\n",err);

  return 0;
}
