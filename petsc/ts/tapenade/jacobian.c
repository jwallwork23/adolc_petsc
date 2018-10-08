#include "ex16adj_d.h"

void ComputeJacobian(double f[],const double *x,const double mu,double J[2][2]) {
    int i,j;
    unsigned int m = 2; 	// TODO: wrap array in struct to generalise
    unsigned int n = 2;
    double fd[m];
    double seed[n];
    for(i=0; i<n; i++){
        seed[i] = 1;
        rhs_d(f,fd,x,seed,mu);
        seed[i] = 0;
        for(j=0; j<m; j++){
            J[j][i] = fd[j];
    }
  }
}
