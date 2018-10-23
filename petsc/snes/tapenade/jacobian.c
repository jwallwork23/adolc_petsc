#include "ex1_d.h"

void ComputeJacobian1(double ff[2],const double *xx, double J[4]) {
    int i,j;
    double ffd[2];
    double seed[2];
    for(i=0; i<2; i++){
        seed[i] = 1;
        f1_d(ff,ffd,xx,seed);
        seed[i] = 0;
        for(j=0; j<2; j++){
            J[2*j+i] = ffd[j];
        }
    }
}

void ComputeJacobian2(double ff[2],const double *xx, double J[4]) {
    int i,j;
    double ffd[2];
    double seed[2];
    for(i=0; i<2; i++){
        seed[i] = 1;
        f2_d(ff,ffd,xx,seed);
        seed[i] = 0;
        for(j=0; j<2; j++){
            J[2*j+i] = ffd[j];
        }
    }
}

/*
void ComputeJacobian(double ff[2], const double *xx, double J[2][2]) {
    int i,j;
    double ffd[2];
    double seed[2];
    for(i=0; i<2; i++){
        seed[i] = 1;
        f1_d(ff,ffd,xx,seed);
        seed[i] = 0;
        for(j=0; j<2; j++){
            J[j][i] = ffd[j];
    }
  }
}
*/
