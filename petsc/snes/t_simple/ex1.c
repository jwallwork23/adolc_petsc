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
  printf("x = [%.4f, %.4f]\n",xx[0],xx[1]);
  printf("f(x) = [%.4f, %.4f]\n",ff[0],ff[1]);

  // derivative calculation
  double ffd[2];
  f1_d(ff,ffd,xx,xx);
  printf("f'(x) = [%.4f, %.4f]\n",ffd[0],ffd[1]);

  // Jacobian calculation
  double J[2][2];
  double seed[2];	// rows of identity matrix
  for(i=0; i<2; i++){
    seed[i] = 1;
    f1_d(ff,ffd,xx,seed);
    seed[i] = 0;
    for(j=0; j<2; j++){
      J[j][i] = ffd[j];
    }
  }

  // print Jacobian
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
