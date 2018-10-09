//#include <petscdm.h>

void RHSLocal(double **u,int xs,int ys,int xm,int ym,int Mx,int My,double sx,double sy,double **f)
{
  int    i,j;
  double two = 2.0,uxx,uyy;

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      if (i == 0 || j == 0 || i == Mx-1 || j == My-1) {
        f[j][i] = u[j][i];
        continue;
      }
      uxx     = (-two*u[j][i] + u[j][i-1] + u[j][i+1])*sx;
      uyy     = (-two*u[j][i] + u[j-1][i] + u[j+1][i])*sy;
      f[j][i] = uxx + uyy;
    }
  }
}
