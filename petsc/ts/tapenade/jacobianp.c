#include "ex16adjp_d.h"

void ComputeJacobianP(double f[],const double *x,const double mu,double J[2][1]) {
    int i,j;
    unsigned int m = 2;
    unsigned int n = 1;		// TODO: wrap array in struct to generalise
    double fd[m];
    double seed = 1;
    for(i=0; i<n; i++){
        rhsp_d(f,fd,x,mu,seed);
        for(j=0; j<m; j++){
            J[i][j] = fd[j];
    }
  }
}
