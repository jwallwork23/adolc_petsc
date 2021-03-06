#include <stdbool.h>
#include <string.h>
#include <stdio.h>

$input int m;
$assume(m<SIZELIMIT && m>0);
$input bool transa, transb;
$input double alpha, beta, alphad, betad;
$input double A[m][m], B[m][m], Cin[m][m];
$input double Cdin[m][m];
$output double Cout[m][m];
$output double Cdout[m][m];

#include "../derivatives/mxm_d.c"

void main() {
    double C[m][m], Cd[m][m];
    memcpy(&C[0][0], &Cin[0][0], m*m*sizeof(double));
    memcpy(&Cd[0][0], &Cdin[0][0], m*m*sizeof(double));
    extra_naive_dgemm_dscalar(transa, transb, m, alpha, alphad, A, B, beta, betad, C, Cd);
    memcpy(&Cout[0][0], &C[0][0], m*m*sizeof(double));
    memcpy(&Cdout[0][0], &Cd[0][0], m*m*sizeof(double));
}
