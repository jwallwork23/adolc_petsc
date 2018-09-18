
static char help[] = "Maps global vector indices to local, and back again.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt         M = 4,N = 3,i,j,k = 0,n,xs,ys,xm,ym;
  PetscErrorCode   ierr;
  DM               da;
  Vec              local,global,scaling;
  DMBoundaryType   bx    = DM_BOUNDARY_PERIODIC,by = DM_BOUNDARY_PERIODIC;
  DMDAStencilType  stype = DMDA_STENCIL_STAR;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = VecDuplicate(global,&scaling);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  /* Insert indices into global vector and take note of boundary points */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  for (j=ys; j<ym; j++) {
    for (i=xs; i<xm; i++) {
      n = 2;
      if ((i == xs) || (i == xm-1)) n++;
      if ((j == ys) || (j == ym-1)) n++;
      ierr = VecSetValue(scaling,k,n,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecSetValue(global,k,k+1,INSERT_VALUES);CHKERRQ(ierr);
      k++;
    }
  }
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Map into local space */
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = VecView(local,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Map back to global space and scale indices */
  ierr = DMLocalToGlobalBegin(da,local,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = VecPointwiseDivide(global,global,scaling);CHKERRQ(ierr);
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free memory */
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&scaling);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}



/*TEST
   
   test:
      requires: x
      nsize: 2
      args: -nox

TEST*/
