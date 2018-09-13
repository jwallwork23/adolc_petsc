static char help[] = "Playing around with MatColoring functions.\n\n"

#include <petscsnes.h>

int main(int argc,char **args)
{
  ISColoring     iscoloring;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  ierr = PetscFinalize();

  return ierr;
}
