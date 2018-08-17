#include "ex1_d.h"

void J1(double ff[2], double ffd[2], const double *xx, const double *xxd, double J[2][2]) {
    int i,j;
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
