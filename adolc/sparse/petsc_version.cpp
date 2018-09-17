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

  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;

  PetscInt n=6,m=3,i,j;
  PetscScalar x[n],c[m];
  adouble xad[n],cad[m];

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

  PetscScalar **Jdense;
  Jdense = myalloc2(m,n);

  jacobian(tag,m,n,x,Jdense);

  ierr = PrintMat(" J",m,n,Jdense);CHKERRQ(ierr);
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

/*--------------------------------------------------------------------------*/
/*                                                     preallocate nonzeros */
/*--------------------------------------------------------------------------*/

  PetscInt        *dnz,*onz,*cols;

  ierr = MatPreallocateInitialize(PETSC_COMM_WORLD,m,n,dnz,onz);CHKERRQ(ierr);
  for (i=0;i<m;i++) {
    ierr = PetscMalloc1(JP[i][0],&cols);CHKERRQ(ierr);
    for (j=1;j<= (int) JP[i][0];j++) {
      cols[j-1] = JP[i][j];
    }
    ierr = MatPreallocateSet(i,n,cols,dnz,onz);CHKERRQ(ierr);
    ierr = PetscFree(cols);CHKERRQ(ierr);
  }
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  // TODO: If it turns out this isn't doing the right job then try MatCreateSeqAIJ with nz and nnz

/*--------------------------------------------------------------------------*/
/*                                                            colour matrix */
/*--------------------------------------------------------------------------*/

  ISColoring      iscoloring;
  MatColoring     coloring;
  MatFDColoring   fdcoloring;
  Mat             J;

  // Create Jacobian object, assemnling with preallocated nonzeros
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // Colour Jacobian
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Creating colouring of J...\n");
  ierr = MatColoringCreate(J,&coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);      // Use 'smallest last' method
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring,&iscoloring);CHKERRQ(ierr);
  ierr = MatFDColoringCreate(J,iscoloring,&fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetUp(J,iscoloring,fdcoloring);CHKERRQ(ierr);


/****************************************************************************/
/*******       free workspace and finalise                    ***************/
/****************************************************************************/

  ierr = MatFDColoringDestroy(&fdcoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  for (i=0;i<m;i++)
    free(JP[i]);
  free(JP);
  myfree2(Jdense);

  ierr = PetscFinalize();

  return ierr;
}

PetscErrorCode PrintMat(const char* name,PetscInt m,PetscInt n,PetscScalar **M)
{
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
