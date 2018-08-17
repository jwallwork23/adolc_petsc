#include "ex16adj_d.h"

void ComputeJacobian(double f[2],const double *x,const double mu,double J[2][2]) {
    int i,j;
    double fd[2];
    double seed[2];
    for(i=0; i<2; i++){
        seed[i] = 1;
        rhs_d(f,fd,x,seed,mu);
        seed[i] = 0;
        for(j=0; j<2; j++){
            J[j][i] = fd[j];
    }
  }
}
