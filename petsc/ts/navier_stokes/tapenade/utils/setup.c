#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>
#include "mms.c"
#include "residuals.c"
#include "../../../utils/init.cpp"


typedef struct {
  PetscInt          dim;
  PetscBool         simplex;
  PetscInt          mms;
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  options->mms     = 1;

  ierr = PetscOptionsBegin(comm, "", "Navier-Stokes Equation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex46.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex46.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mms", "The manufactured solution to use", "ex46.c", options->mms, &options->mms, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateBCLabel(DM dm, const char name[])
{
  DMLabel        label;
  PetscErrorCode ierr; 
  
  PetscFunctionBeginUser;
  ierr = DMCreateLabel(dm, name);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, name, &label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(dm, 1, label);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(dm, label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *ctx)
{
  DM             pdm = NULL;
  const PetscInt dim = ctx->dim;
  PetscBool      hasLabel;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexCreateBoxMesh(comm, dim, ctx->simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  /* If no boundary marker exists, mark the whole boundary */
  ierr = DMHasLabel(*dm, "marker", &hasLabel);CHKERRQ(ierr);
  if (!hasLabel) {ierr = CreateBCLabel(*dm, "marker");CHKERRQ(ierr);}
  /* Distribute mesh over processes */
  ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
  if (pdm) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = pdm;
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MonitorError(TS ts, PetscInt step, PetscReal crtime, Vec u, void *ctx)
{
  AppCtx        *user = (AppCtx *) ctx;
  DM             dm;
  PetscReal      ferrors[2];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMComputeL2FieldDiff(dm, crtime, user->exactFuncs, NULL, u, ferrors);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Timestep: %04d time = %-8.4g \t L_2 Error: [%2.3g, %2.3g]\n", (int) step, (double) crtime, (double) ferrors[0], (double) ferrors[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
