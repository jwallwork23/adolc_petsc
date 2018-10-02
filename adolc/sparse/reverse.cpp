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
  MPI_Comm        comm = MPI_COMM_WORLD;

  ierr = PetscInitialize(&argc,&args,(char*)0,NULL);if (ierr) return ierr;

  PetscInt n = 6,m = 3,i,j;
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

  ierr = PetscPrintf(comm,"\n c = ");CHKERRQ(ierr);
  for(j=0;j<m;j++)
      ierr = PetscPrintf(comm," %e ",c[j]);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);

/****************************************************************************/
/********           For comparisons: Full Jacobian                   ********/
/****************************************************************************/

  PetscScalar **Jdense;
  Jdense = myalloc2(m,n);

  jacobian(tag,m,n,x,Jdense);

  ierr = PrintMat(comm," Jacobian:",m,n,Jdense);CHKERRQ(ierr);

/****************************************************************************/
/*******            reverse mode with matrix assembly         ***************/
/****************************************************************************/

/*--------------------------------------------------------------------------*/
/*                                                sparsity pattern Jacobian */
/*--------------------------------------------------------------------------*/

  unsigned int  **JP = NULL;                /* compressed block row storage */
  PetscInt      ctrl[3] = {0,0,0};

  JP = (unsigned int **) malloc(m*sizeof(unsigned int*));
  jac_pat(tag,m,n,x,JP,ctrl);

  ierr = PetscPrintf(comm," Sparsity pattern: \n");CHKERRQ(ierr);
  for (i=0;i<m;i++) {
    ierr = PetscPrintf(comm," %d: ",i);CHKERRQ(ierr);
    for (j=1;j<= (int) JP[i][0];j++)
      ierr = PetscPrintf(comm," %d ",JP[i][j]);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr);
  }

/*--------------------------------------------------------------------------*/
/*                                                     preallocate nonzeros */
/*--------------------------------------------------------------------------*/

  Mat             J;
  PetscInt        k,nnz[m];
  PetscScalar     one = 1.;

  // Get number of nonzeros per row
  for (i=0; i<m; i++)
    nnz[i] = (PetscInt) JP[i][0];

  // Create Jacobian object, assembling with preallocated nonzeros as ones
  ierr = MatCreateSeqAIJ(comm,m,n,0,nnz,&J);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=1; j<=nnz[i]; j++) {
      k = JP[i][j];
      ierr = MatSetValues(J,1,&i,1,&k,&one,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

/*--------------------------------------------------------------------------*/
/*                                                            trace forward */
/*--------------------------------------------------------------------------*/

  PetscInt        q = m;
  PetscScalar     *f;

  ierr = PetscMalloc1(m,&f);CHKERRQ(ierr);

  zos_forward(tag,m,n,1,x,c);
  ierr = PetscPrintf(comm,"\n c = ");CHKERRQ(ierr);
  for(j=0;j<m;j++)
      ierr = PetscPrintf(comm," %e ",c[j]);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);

/*--------------------------------------------------------------------------*/
/*                                                        back out Jacobian */
/*--------------------------------------------------------------------------*/

  PetscScalar **I,**Jrev;

  I = myallocI2(q);
  Jrev = myalloc2(q,n);
  fov_reverse(tag,m,n,q,I,Jrev);
  ierr = PrintMat(PETSC_COMM_WORLD," Jacobian by reverse mode:",q,n,Jrev);CHKERRQ(ierr);

/*--------------------------------------------------------------------------*/
/*                          Jacobian transpose vector product, the long way */
/*--------------------------------------------------------------------------*/

// TODO

/****************************************************************************/
/*******            reverse mode matrix free                  ***************/
/****************************************************************************/

// TODO

/****************************************************************************/
/*******       free workspace and finalise                    ***************/
/****************************************************************************/

  ierr = PetscFree(f);CHKERRQ(ierr);
  myfree2(Jrev);
  myfreeI2(q,I);
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
    for(j=0; j<n ;j++)
      ierr = PetscPrintf(comm," %10.4f ", M[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);
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
