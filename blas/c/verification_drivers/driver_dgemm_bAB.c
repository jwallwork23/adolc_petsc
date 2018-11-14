#include <stdbool.h>
#include <string.h>
#include <stdio.h>

$input int m;
$assume(m<SIZELIMIT && m>0);
$input bool transa, transb;
$input double alpha, beta;
$input double A[m][m], B[m][m], Cin[m][m];
$output double Cout[m][m];
$input double Abin[m][m], Bbin[m][m], Cbin[m][m];
$output double About[m][m], Bbout[m][m], Cbout[m][m];

#include "../derivatives/mxm_b.c"

void main() {
    double C[m][m];
    double Ab[m][m], Bb[m][m], Cb[m][m];
    memcpy(&C[0][0], &Cin[0][0], m*m*sizeof(double));
    memcpy(&Ab[0][0], &Abin[0][0], m*m*sizeof(double));
    memcpy(&Bb[0][0], &Bbin[0][0], m*m*sizeof(double));
    memcpy(&Cb[0][0], &Cbin[0][0], m*m*sizeof(double));

    naive_dgemm_bAB(transa, transb, m, alpha, A, Ab, B, Bb, beta, C, Cb);

    memcpy(&Cout[0][0], &C[0][0], m*m*sizeof(double));
    memcpy(&About[0][0], &Ab[0][0], m*m*sizeof(double));
    memcpy(&Bbout[0][0], &Bb[0][0], m*m*sizeof(double));
    memcpy(&Cbout[0][0], &Cb[0][0], m*m*sizeof(double));
}
