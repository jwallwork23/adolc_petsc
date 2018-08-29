#include <petscts.h>
#include <adolc/adolc.h>
#include "utils.c"

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble u,v;
} aField;

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  Field           **input,**output;
  adouble         *input_a = new adouble[8];
  adouble         *output_a = new adouble[8];
  PetscInt        i,j,k=0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  PetscFunctionBeginUser;

  // Allocate memory for Fields and set some values
  ierr = PetscMalloc1(2,&input);
  ierr = PetscMalloc1(2,&output);
  for(j=0;j<2;j++){
    ierr = PetscMalloc1(2,&input[j]);
    ierr = PetscMalloc1(2,&output[j]);
    for(i=0;i<2;i++){
      input[j][i].u = 2*j+i+1;
      input[j][i].v = -(2*j+i+1);
    }
  }

  trace_on(1);

  for(j=0;j<2;j++){
    for(i=0;i<2;i++){
      input_a[k]    <<= input[j][i].u;
      output_a[k]   =   input_a[k];
      output_a[k++] >>= output[j][i].u;
    }
  }
  trace_off();

  // Free memory
  ierr = PetscFree(input);CHKERRQ(ierr);
  ierr = PetscFree(output);CHKERRQ(ierr);

  delete[] input_a;
  delete[] output_a;

  ierr = PetscFinalize();
  return ierr;
}
