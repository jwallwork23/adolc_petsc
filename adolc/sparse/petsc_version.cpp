#include <petscsnes.h>
#include <adolc/adolc.h>
#include <adolc/adolc_sparse.h>

#define tag 1

PetscErrorCode PrintMat(MPI_Comm comm,const char* name,PetscInt n,PetscInt m,PetscScalar **M);
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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n c = ");CHKERRQ(ierr);
  for(j=0;j<m;j++)
      ierr = PetscPrintf(PETSC_COMM_WORLD," %e ",c[j]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

/****************************************************************************/
/********           For comparisons: Full Jacobian                   ********/
/****************************************************************************/

  PetscScalar **Jdense;
  Jdense = myalloc2(m,n);

  jacobian(tag,m,n,x,Jdense);

  ierr = PrintMat(PETSC_COMM_WORLD," J",m,n,Jdense);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

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

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSparsity pattern of Jacobian: \n");CHKERRQ(ierr);
  for (i=0;i<m;i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD," %d: ",i);CHKERRQ(ierr);
    for (j=1;j<= (int) JP[i][0];j++)
      ierr = PetscPrintf(PETSC_COMM_WORLD," %d ",JP[i][j]);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);

/*--------------------------------------------------------------------------*/
/*                                                     preallocate nonzeros */
/*--------------------------------------------------------------------------*/
/*
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
*/
  // TODO: try preallocating using MatCreateSeqAIJ with nz and nnz

/*--------------------------------------------------------------------------*/
/*                                      obtain a colouring for the Jacobian */
/*--------------------------------------------------------------------------*/

  ISColoring      iscoloring;
  MatColoring     coloring;
  Mat             J;
  PetscInt        k,max=(int) JP[0][0],nis=0,size;
  PetscScalar     one=1.,**Seed = NULL;
  IS              *isp,is;
  const PetscInt  *nindices;

  // Create Jacobian object, assembling with preallocated nonzeros as ones
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  for (i=0;i<m;i++) {
    if ((int) JP[i][0] > max)
      nis = JP[i][0];
    for (j=1;j<= (int) JP[i][0];j++) {
      k = JP[i][j];
      ierr = MatSetValues(J,1,&i,1,&k,&one,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // Colour Jacobian
  ierr = MatColoringCreate(J,&coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);      // Use 'smallest last' method
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring,&iscoloring);CHKERRQ(ierr);
  //ierr = MatColoringView(coloring,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = ISColoringView(iscoloring,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

/*--------------------------------------------------------------------------*/
/*                                                              seed matrix */
/*--------------------------------------------------------------------------*/

  Seed = myalloc2(n,max);

  ierr = ISColoringGetIS(iscoloring,&nis,&isp);CHKERRQ(ierr);
  for (i=0;i<max;i++) {
    is = *(isp+i);
    //ierr = ISView(is,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is,&size);CHKERRQ(ierr);
    ierr = ISGetIndices(is,&nindices);CHKERRQ(ierr);
    for (j=0;j<size;j++)
      Seed[nindices[j]][i] = 1.;  // Extract nonzeros for seed matrix
    ierr = ISRestoreIndices(is,&nindices);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&isp);CHKERRQ(ierr);
  ierr = PrintMat(PETSC_COMM_WORLD,"Seed matrix:",n,max,Seed);CHKERRQ(ierr);
  myfree2(Seed);


/****************************************************************************/
/*******       free workspace and finalise                    ***************/
/****************************************************************************/

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

PetscErrorCode PrintMat(MPI_Comm comm,const char* name,PetscInt m,PetscInt n,PetscScalar **M)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"%s \n",name);CHKERRQ(ierr);
  for(i=0; i<m ;i++) {
    ierr = PetscPrintf(comm,"\n %d: ",i);CHKERRQ(ierr);
    for(j=0;j<n ;j++)
      ierr = PetscPrintf(comm," %10.4f ", M[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PassiveEvaluate(PetscScalar *x,PetscScalar *c)
{
  PetscFunctionBegin;
  c[0] = 2*x[0]+x[1]-2.0;
  c[0] += PetscCosScalar(x[3])*PetscSinScalar(x[4]);
  c[1] = x[2]*x[2]+x[3]*x[3]-2.0;
  c[2] = 3*x[4]*x[5] - 3.0+PetscSinScalar(x[4]*x[5]);
  //c[3] = x[3];c[4] = x[4];c[5] = x[5]; // Trivial extension to a square problem
  PetscFunctionReturn(0);
}

PetscErrorCode ActiveEvaluate(adouble *x,adouble *c)
{
  PetscFunctionBegin;
  c[0] = 2*x[0]+x[1]-2.0;
  c[0] += PetscCosScalar(x[3])*PetscSinScalar(x[4]);
  c[1] = x[2]*x[2]+x[3]*x[3]-2.0;
  c[2] = 3*x[4]*x[5] - 3.0+PetscSinScalar(x[4]*x[5]);
  //c[3] = x[3];c[4] = x[4];c[5] = x[5]; // Trivial extension to a square problem
  PetscFunctionReturn(0);
}
