/*
  Compute Jacobian using naive approach of propagating canonical Cartesian basis through the forward mode of AD.
*/
void ComputeJacobian(double **f,double **u,int gxs,int gxm,int gys,int gym,double sx,double sy,int My,double **J){

  int    i,j,l,k,n = gxm*gym;
  double fd[n],seed[n];

  for (j=gys; j<gym; j++) {
    for (i=gxs; i<gxm; i++) {
      seed[j][i] = 1.;
      RHSLocal_d(u,seed,gxs,gxm,gys,gym,sx,syf,fd);
      seed[j][i] = 0.;
      for (l=gys; l<gym; l++) {
        for (k=gxs; k<gxm; k++) {
          J[l][k] = fd[l][k];
        }
      }
    }
  }
}

