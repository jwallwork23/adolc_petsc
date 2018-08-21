#include <adolc/adolc.h>
#include <stdio.h>


void f1(double ff[2],double *xx){
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
  // initialisation
  int i,j;
  double ff[2];
  double xx[2] = {2., 1.};

  // function evaluation
  f1(ff,xx);
  printf("x = [%.4f, %.4f]\n",xx[0],xx[1]);
  printf("f(x) = [%.4f, %.4f]\n",ff[0],ff[1]);

  // Jacobian calculation
  double** J = (double**) malloc(2*sizeof(double*));	/* Note different */
  for(i=0;i<2;i++)					/*   form than in */
    J[i] = (double*)malloc((i+1)*sizeof(double));	/*     tapenade   */
  jacobian(1,2,2,xx,J);

  // derivative calculation
  double ffd[2];
  ffd[0] = 0.; ffd[1] = 0.;
  for(i=0; i<2; i++){
    for(j=0; j<2; j++){
      ffd[i] += J[i][j] * xx[j];
    }
  }
  printf("f'(x) = [%.4f, %.4f]\n",ffd[0],ffd[1]);

  // print Jacobian
  printf("J = \n");
  for(i=0;i<2;i++){
    printf("    [");
    for(j=0; j<2; j++){
      printf("%.4f, ",J[i][j]);
    }
    printf("]\n");
  }
  free(J);

  return 0;
}

