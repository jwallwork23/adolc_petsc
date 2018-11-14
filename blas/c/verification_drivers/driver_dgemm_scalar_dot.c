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
$assume(transa && transb && alphad == 2 && betad == 2);

#include "../derivatives/byhand.c"

void main() {
    double C[m][m], Cd[m][m];
    memcpy(&C[0][0], &Cin[0][0], m*m*sizeof(double));
    memcpy(&Cd[0][0], &Cdin[0][0], m*m*sizeof(double));
    naive_dgemm_scalar_dot(transa, transb, m, alpha, alphad, A, B, beta, betad, C, Cd);
    memcpy(&Cout[0][0], &C[0][0], m*m*sizeof(double));
    memcpy(&Cdout[0][0], &Cd[0][0], m*m*sizeof(double));
    int i, j;
    for(i=0;i<m;i++) {
        for(j=0;j<m;j++) {
            printf("%f\t %f\t %f\t %f\t %f\n",alpha,alphad,beta,betad,Cd[i][j]);
        }
    }
}
