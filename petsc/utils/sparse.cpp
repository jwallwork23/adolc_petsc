#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

/*
  Print matrices involved in sparse computations.
*/
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

/*
  Print sparsity pattern
*/
PetscErrorCode PrintSparsity(MPI_Comm comm,PetscInt m,unsigned int **JP)
{
  PetscErrorCode ierr;
  PetscInt       i,j;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"Sparsity pattern:\n");CHKERRQ(ierr);
  for(i=0; i<m ;i++) {
    ierr = PetscPrintf(comm,"\n %2d: ",i);CHKERRQ(ierr);
    for(j=1; j<= (PetscInt) JP[i][0] ;j++)
      ierr = PetscPrintf(comm," %2d ", JP[i][j]);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(comm,"\n\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode GetColoring(DM da,PetscInt m,PetscInt n,unsigned int **JP,ISColoring *iscoloring)
{
  PetscErrorCode         ierr;
  Mat                    S;
  MatColoring            coloring;
  PetscInt               i,j,k,nnz[m],onz[m];
  PetscScalar            one = 1.;

  PetscFunctionBegin;

  /*
    Extract number of nonzeros and colours required from JP.
  */
  for (i=0; i<m; i++) {
    nnz[i] = (PetscInt) JP[i][0];
    onz[i] = nnz[i];
    for (j=1; j<=nnz[i]; j++) {
      if (i == (PetscInt) JP[i][j])
        onz[i]--;
    }
  }

  /*
     Preallocate nonzeros as ones. 

     NOTE: Using DMCreateMatrix overestimates nonzeros.
     FIXME: I think it is probably this PETSC_COMM_SELF which is causing trouble. Use MatSetValuesLocal
  */
  ierr = MatCreateAIJ(PETSC_COMM_SELF,m,n,PETSC_DETERMINE,PETSC_DETERMINE,0,nnz,0,onz,&S);CHKERRQ(ierr);
  ierr = MatSetFromOptions(S);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=1; j<=nnz[i]; j++) {
      k = JP[i][j];
      ierr = MatSetValues(S,1,&i,1,&k,&one,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
    Extract colouring, with smallest last ('sl') as default.

    NOTE: Use -mat_coloring_type <sl,lf,id,natural,greedy,jp> to change mode.
    FIXME: only natural is currently working if BCs are considered
    FIXME: jp and greedy are not currently working at all
  */
  ierr = MatColoringCreate(S,&coloring);CHKERRQ(ierr);
  ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);
  ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
  ierr = MatColoringApply(coloring,iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode CountColors(ISColoring iscoloring,PetscInt *p)
{
  PetscErrorCode ierr;
  IS             *is;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,p,&is);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring,PetscScalar **Seed)
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
      Seed[indices[j]][i] = 1.;
    ierr = ISRestoreIndices(is[i],&indices);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&is);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode GetRecoveryMatrix(PetscScalar **Seed,unsigned int **JP,PetscInt m,PetscInt p,PetscScalar **Rec)
{
  PetscInt i,j,k,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0; colour<p; colour++) {
      Rec[i][colour] = -1.;
      for (k=1; k<=(PetscInt) JP[i][0]; k++) {
        j = (PetscInt) JP[i][k];
        if (Seed[j][colour] == 1.) {
          Rec[i][colour] = j;
          break;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RecoverJacobian(Mat J,PetscInt m,PetscInt p,PetscScalar **Rec,PetscScalar **Jcomp)
{
  PetscErrorCode ierr;
  PetscInt       i,j,colour;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (colour=0; colour<p; colour++) {
      j = (PetscInt) Rec[i][colour];
      if (j != -1)
        ierr = MatSetValuesLocal(J,1,&i,1,&j,&Jcomp[i][colour],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

