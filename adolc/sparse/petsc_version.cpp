#include <petscsnes.h>
#include <adolc/adolc.h>
#include <adolc/adolc_sparse.h>

#define tag 1

PetscErrorCode PrintMat(const char* name,PetscInt n,PetscInt m,PetscScalar **M);
PetscErrorCode PassiveEvaluate(PetscScalar *x,PetscScalar *c);
PetscErrorCode ActiveEvaluate(adouble *x,adouble *c);

int main(int argc,char **args)
{
  PetscErrorCode  ierr;
  ISColoring      iscoloring;
  MatColoring     coloring;
  MatFDColoring   fdcoloring;

  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;

  PetscInt n=6,m=3,i,j;
  PetscScalar x[6],c[3];
  adouble xad[6],cad[3];

/****************************************************************************/
/*******                function evaluation                   ***************/
/****************************************************************************/

  for(i=0;i<n;i++)
    x[i] = log(1.0+i);

  /* Tracing of function c(x) */
  trace_on(tag);
    for(i=0;i<n;i++)
      xad[i] <<= x[i];

    ierr = ActiveEvaluate(xad,cad);CHKERRQ(ierr);

    for(i=0;i<m;i++)
      cad[i] >>= c[i];
  trace_off();

  printf("\n c =  ");
  for(j=0;j<m;j++)
      printf(" %e ",c[j]);
  printf("\n");

/****************************************************************************/
/********           For comparisons: Full Jacobian                   ********/
/****************************************************************************/

  PetscScalar **J;
  J = myalloc2(m,n);

  jacobian(tag,m,n,x,J);

  ierr = PrintMat(" J",m,n,J);CHKERRQ(ierr);
  printf("\n");

/****************************************************************************/
/*******       sparse Jacobians, separate drivers             ***************/
/****************************************************************************/

/*--------------------------------------------------------------------------*/
/*                                                sparsity pattern Jacobian */
/*--------------------------------------------------------------------------*/

  unsigned int  **JP=NULL;                /* compressed block row storage */
  PetscInt ctrl[3];

  JP = (unsigned int **) malloc(m*sizeof(unsigned int*));
  ctrl[0] = 0;
  ctrl[1] = 0;
  ctrl[2] = 0;

  jac_pat(tag, m, n, x, JP, ctrl);

  printf("\n");
  printf("Sparsity pattern of Jacobian: \n");
  for (i=0;i<m;i++) {
    printf(" %d: ",i);
    for (j=1;j<= (int) JP[i][0];j++)
      printf(" %d ",JP[i][j]);
      printf("\n");
  }
  printf("\n");

/****************************************************************************/
/*******       free memory and finalise                       ***************/
/****************************************************************************/

  for (i=0;i<m;i++)
    free(JP[i]);
  free(JP);
  myfree2(J);

  ierr = PetscFinalize();

  return ierr;
}

PetscErrorCode PrintMat(const char* name,PetscInt m,PetscInt n,PetscScalar **M)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;

  printf("%s \n",name);
  for(i=0; i<m ;i++) {
      printf("\n %d: ",i);
      for(j=0;j<n ;j++)
          printf(" %10.4f ", M[i][j]);
  }
  printf("\n");

  PetscFunctionReturn(0);
}

PetscErrorCode PassiveEvaluate(PetscScalar *x,PetscScalar *c)
{
  PetscFunctionBegin;
  c[0] = 2*x[0]+x[1]-2.0;
  c[0] += PetscCosScalar(x[3])*PetscSinScalar(x[4]);
  c[1] = x[2]*x[2]+x[3]*x[3]-2.0;
  c[2] = 3*x[4]*x[5] - 3.0+PetscSinScalar(x[4]*x[5]);
  PetscFunctionReturn(0);
}

PetscErrorCode ActiveEvaluate(adouble *x,adouble *c)
{
  PetscFunctionBegin;
  c[0] = 2*x[0]+x[1]-2.0;
  c[0] += PetscCosScalar(x[3])*PetscSinScalar(x[4]);
  c[1] = x[2]*x[2]+x[3]*x[3]-2.0;
  c[2] = 3*x[4]*x[5] - 3.0+PetscSinScalar(x[4]*x[5]);
  PetscFunctionReturn(0);
}
