#include <adolc/adolc.h>
#include <stdio.h>

void FormFunction1(double ff[2],double *xx){

  adouble ff_a[2];
  adouble *xx_a = new adouble[2];

  trace_on(1);

  xx_a[0] <<= xx[0];
  xx_a[1] <<= xx[1];

  ff_a[0] = xx_a[0]*xx_a[0] + xx_a[0]*xx_a[1] - 3.0;
  ff_a[1] = xx_a[0]*xx_a[1] + xx_a[1]*xx_a[1] - 6.0;

  ff_a[0] >>= ff[0];
  ff_a[1] >>= ff[1];

  delete[] xx_a;
  trace_off(1);

}



int main()
{
  int i;
  double ff[2];
  double *xx = new double[2];

  xx[0] = 2.;
  xx[1] = -1.;

  FormFunction1(ff,xx);

  printf("x = [%f, %f]\n",xx[0],xx[1]);
  printf("f(x) = [%f, %f]\n",ff[0],ff[1]);

  double** J = (double**) malloc(2*sizeof(double*));
  for(i=0;i<2;i++)
    J[i] = (double*)malloc((i+1)*sizeof(double));
  jacobian(1,2,2,xx,J);

  printf("J = \n");
  for(i=0;i<2;i++){
    printf("    [");
    for(int j=0; j<2; j++){
      printf("%.4f, ",J[i][j]);
    }
    printf("]\n");
  }

  return 0;
}

