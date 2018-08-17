#include <stdio.h>
#include "ex1_d.h"

int main(){

  // initialisation
  int i,j;
  double ff[2];
  double xx[2];
  xx[0] = 2.;
  xx[1] = -1;

  // function evaluation
  f1(ff,xx);

  // derivative calculation
  double ffd[2];
  f1_d(ff,ffd,xx,xx);

  // Jacobian calculation
  double J[2][2];
  J1(ff,ffd,xx,xx,J);

  // print results
  printf("x = [%.4f, %.4f]\n",xx[0],xx[1]);
  printf("f(x) = [%.4f, %.4f]\n",ff[0],ff[1]);
  printf("f'(x) = [%.4f, %.4f]\n",ffd[0],ffd[1]);
  printf("J =\n");
  for(i=0; i<2; i++){
    printf("    [");
    for(j=0; j<2; j++){
      printf("%.4f, ",J[i][j]);
    }
    printf("]\n");
  }

  return 0;
}

