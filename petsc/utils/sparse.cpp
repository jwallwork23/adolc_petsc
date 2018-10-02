#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

/*@C
  Simple matrix printing

  Input parameters:
  comm - MPI communicator
  name - name of matrix to print
  m,n  - number of rows and columns, respectively
  M    - matrix to print
@*/
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

/*@C
  Print sparsity pattern

  Input parameters:
  comm     - MPI communicator
  m        - number of rows

  Output parameter:
  sparsity - matrix sparsity pattern, typically computed using an ADOL-C function such as jac_pat or
  hess_pat
@*/
PetscErrorCode PrintSparsity(MPI_Comm comm,PetscInt m,unsigned int **sparsity)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"Sparsity pattern:\n");CHKERRQ(ierr);
  for(i=0; i<m ;i++) {
    ierr = PetscPrintf(comm,"\n %2d: ",i);CHKERRQ(ierr);
    for(j=1; j<= (PetscInt) sparsity[i][0] ;j++)
      ierr = PetscPrintf(comm," %2d ",sparsity[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  Extract an index set coloring from a sparsity pattern

  Input parameters:
  da         - distributed array
  m,n        - number of rows and columns, respectively
  sparsity   - matrix sparsity pattern, typically computed using an ADOL-C function such as jac_pat
  or hess_pat

  Output parameter:
  iscoloring - index set coloring corresponding to the sparsity pattern under the given coloring type

  Notes:
  Use -mat_coloring_type <sl,lf,id,natural,greedy,jp> to change coloring type used
  FIXME: only natural is currently working in serial case if BCs are considered
  FIXME: jp and greedy run in parallel case, but give Jacobians leading to divergence
@*/
PetscErrorCode GetColoring(DM da,PetscInt m,PetscInt n,unsigned int **sparsity,ISColoring *iscoloring)
{
  PetscErrorCode         ierr;
  Mat                    P;		/* Mat containing nonzero entries */
  MatColoring            coloring;
  PetscInt               i,j,k,xproc,yproc,zproc,nnz[m],onz[m];
  PetscScalar            one = 1.;
  //ISLocalToGlobalMapping ltog;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,&xproc,&yproc,&zproc,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  /*
    Extract number of nonzeros and off-diagonal nonzeros from sparsity pattern.
  */
  for (i=0; i<m; i++) {
    nnz[i] = (PetscInt) sparsity[i][0];
    onz[i] = nnz[i];
    for (j=1; j<=nnz[i]; j++) {
      if (i == (PetscInt) sparsity[i][j])
        onz[i]--;
    }
  }

  /*
     Preallocate nonzeros as ones. 

     FIXME: Perhaps consider matrix passed into function from outside
  */
  /* Version which seems most promising */
  ierr = MatCreateAIJ(PETSC_COMM_SELF,m,n,PETSC_DETERMINE,PETSC_DETERMINE,0,nnz,0,onz,&P);CHKERRQ(ierr);

  /* Alternative version */
  //ierr = MatCreateAIJ(PETSC_COMM_WORLD,m,n,PETSC_DETERMINE,PETSC_DETERMINE,0,nnz,0,onz,&P);CHKERRQ(ierr);
  //ierr = DMGetLocalToGlobalMapping(da,&ltog);CHKERRQ(ierr);
  //ierr = MatSetLocalToGlobalMapping(P,ltog,ltog);CHKERRQ(ierr);

  /* Alternative version 2 */
  //ierr = DMCreateMatrix(da,&P);CHKERRQ(ierr);

  ierr = MatSetFromOptions(P);CHKERRQ(ierr);
  ierr = MatSetUp(P);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=1; j<=nnz[i]; j++) {
      k = sparsity[i][j];
      //ierr = MatSetValuesLocal(P,1,&i,1,&k,&one,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(P,1,&i,1,&k,&one,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
    Extract colouring, with smallest last as default.
  */
  //ierr = DMCreateColoring(da,IS_COLORING_GLOBAL,iscoloring);CHKERRQ(ierr);
  //ierr = DMCreateColoring(da,IS_COLORING_LOCAL,iscoloring);CHKERRQ(ierr);

  ierr = MatColoringCreate(P,&coloring);CHKERRQ(ierr);
  if ((xproc > 1) || (yproc > 1) || (zproc > 1)) {
    ierr = MatColoringSetType(coloring,MATCOLORINGJP);CHKERRQ(ierr); // Parallel coloring
  } else {
    ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr); // Serial coloring
  }
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring,iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*@C
  Simple function to count the number of colors used in an index set coloring

  Input parameter:
  iscoloring - the index set coloring to count the number of colors of

  Output parameter:
  p          - number of colors used in iscoloring
@*/
PetscErrorCode CountColors(ISColoring iscoloring,PetscInt *p)
{
  PetscErrorCode ierr;
  IS             *is;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,p,&is);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  Generate a seed matrix defining the partition of columns of a matrix by a particular coloring,
  used for matrix compression

  Input parameter:
  iscoloring - the index set coloring to be used

  Output parameter:
  S          - the resulting seed matrix

  Notes:
  Before calling GenerateSeedMatrix, Seed should be allocated as a logically 2d array with number of
  rows equal to the matrix to be compressed and number of columns equal to the number of colors used
  in iscoloring.
@*/
PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring,PetscScalar **S)
{
  PetscErrorCode ierr;
  IS             *is;
  PetscInt       p,size,i,j;
  const PetscInt *indices;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,&p,&is);CHKERRQ(ierr);
  for (i=0; i<p; i++) {
    ierr = ISGetLocalSize(is[i],&size);CHKERRQ(ierr);
    ierr = ISGetIndices(is[i],&indices);CHKERRQ(ierr);
    for (j=0; j<size; j++)
      S[indices[j]][i] = 1.;
    ierr = ISRestoreIndices(is[i],&indices);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  Establish a look-up matrix whose entries contain the column coordinates of the corresponding entry
  in a matrix which has been compressed using the coloring defined by some seed matrix

  Input parameters:
  S        - the seed matrix defining the coloring
  sparsity - the sparsity pattern of the matrix to be recovered, typically computed using an ADOL-C
             function, such as jac_pat or hess_pat
  m        - the number of rows of Seed (and the matrix to be recovered)
  p        - the number of colors used (also the number of columns in Seed)

  Output parameter:
  R        - the recovery matrix to be used for de-compression
@*/
PetscErrorCode GetRecoveryMatrix(PetscScalar **S,unsigned int **sparsity,PetscInt m,PetscInt p,PetscScalar **R)
{
  PetscInt i,j,k,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0; colour<p; colour++) {
      R[i][colour] = -1.;
      for (k=1; k<=(PetscInt) sparsity[i][0]; k++) {
        j = (PetscInt) sparsity[i][k];
        if (S[j][colour] == 1.) {
          R[i][colour] = j;
          break;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  Recover the values of a sparse matrix from a compressed foramt and insert these into a matrix

  Input parameters:
  m - number of rows of matrix.
  p - number of colors used in the compression of J (also the number of columns of R)
  R - recovery matrix to use in the decompression procedure
  C - compressed matrix to recover values from

  Output parameter:
  A - Mat to be populated with values from compressed matrix
*/
PetscErrorCode RecoverJacobian(Mat A,PetscInt m,PetscInt p,PetscScalar **R,PetscScalar **C)
{
  PetscErrorCode ierr;
  PetscInt       i,j,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0; colour<p; colour++) {
      j = (PetscInt) R[i][colour];
      if (j != -1)
        ierr = MatSetValuesLocal(A,1,&i,1,&j,&C[i][colour],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
