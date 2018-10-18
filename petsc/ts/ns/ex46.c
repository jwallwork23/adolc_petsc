static char help[] = "Time dependent Navier-Stokes problem in 2d and 3d with finite elements.\n\
We solve the Navier-Stokes in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports discretized auxiliary fields (Re) as well as\n\
multilevel nonlinear solvers.\n\
Contributed by: Julian Andrej <juan@tf.uni-kiel.de>\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscts.h>
#include <petscds.h>
#include "mms.c"
#include "residuals.c"
#include "derivatives.c"
// TODO: Create header file ex46.h

/*
  Navier-Stokes equation:

  du/dt + u . grad u - \Delta u - grad p = f
  div u  = 0
*/

typedef struct {
  PetscInt          dim;
  PetscBool         simplex;
  PetscInt          mms;
  PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
} AppCtx;

/* -< \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
static void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal Re    = REYN;
  const PetscInt  Ncomp = dim;
  PetscInt        compI, d;

  for (compI = 0; compI < Ncomp; ++compI) {
    for (d = 0; d < dim; ++d) {
      g3[((compI*Ncomp+compI)*dim+d)*dim+d] = 1.0/Re;
    }
  }
}

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

static PetscErrorCode SetupProblem(PetscDS prob, AppCtx *ctx)
{
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  switch (ctx->mms) {
  case 1:
    ierr = PetscDSSetResidual(prob, 0, f0_mms1_u, f1_u);CHKERRQ(ierr);break;
  case 2:
    ierr = PetscDSSetResidual(prob, 0, f0_mms2_u, f1_u);CHKERRQ(ierr);break;
  }
  ierr = PetscDSSetResidual(prob, 1, f0_p, f1_p);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, g0_uu, g1_uu, NULL,  g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 1, NULL, NULL, g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 1, 0, NULL, g1_pu, NULL,  NULL);CHKERRQ(ierr);
  switch (ctx->dim) {
  case 2:
    switch (ctx->mms) {
    case 1:
      ctx->exactFuncs[0] = mms1_u_2d;
      ctx->exactFuncs[1] = mms1_p_2d;
      break;
    case 2:
      ctx->exactFuncs[0] = mms2_u_2d;
      ctx->exactFuncs[1] = mms2_p_2d;
      break;
    default:
      SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid MMS %D", ctx->mms);
    }
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension %D", ctx->dim);
  }
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) ctx->exactFuncs[0], 1, &id, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *ctx)
{
  DM              cdm = dm;
  const PetscInt  dim = ctx->dim;
  PetscDS         prob;
  PetscFE         fe[2];
  PetscQuadrature q;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, dim, ctx->simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe[0], &q);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, ctx->simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFESetQuadrature(fe[1], q);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 1, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = SetupProblem(prob, ctx);CHKERRQ(ierr);
  while (cdm) {
    PetscObject  pressure;
    MatNullSpace nsp;
    PetscBool    hasLabel;

    ierr = DMGetField(cdm, 1, &pressure);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm(pressure), PETSC_TRUE, 0, NULL, &nsp);CHKERRQ(ierr);
    ierr = PetscObjectCompose(pressure, "nullspace", (PetscObject) nsp);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);

    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
    ierr = DMHasLabel(cdm, "marker", &hasLabel);CHKERRQ(ierr);
    if (!hasLabel) {ierr = CreateBCLabel(cdm, "marker");CHKERRQ(ierr);}
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
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

int main(int argc, char **argv)
{
  AppCtx         ctx;
  DM             dm;
  TS             ts;
  Vec            u, r;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, "petscoptions", help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm, &ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &ctx);CHKERRQ(ierr);
  ierr = PetscMalloc1(2, &ctx.exactFuncs);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, &ctx);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts, MonitorError, &ctx, NULL);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = DMTSSetBoundaryLocal(dm, DMPlexTSComputeBoundary, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIFunctionLocal(dm, DMPlexTSComputeIFunctionFEM, &ctx);CHKERRQ(ierr);
  ierr = DMTSSetIJacobianLocal(dm, DMPlexTSComputeIJacobianFEM, &ctx);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = DMProjectFunction(dm, 0.0, ctx.exactFuncs, NULL, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = TSSolve(ts, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(ctx.exactFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # Full solves
  test:
    suffix: 2d_p2p1_r1
    requires: !single triangle
    filter: sed -e "s~ATOL~RTOL~g" -e "s~ABS~RELATIVE~g"
    args: -dm_refine 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ts_type beuler -ts_max_steps 10 -ts_dt 0.1 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1.0e-10 -fieldsplit_pressure_pc_type jacobi -ksp_monitor_short -ksp_converged_reason -snes_monitor_short -snes_converged_reason -ts_monitor
  test:
    suffix: 2d_p2p1_r2
    requires: !single triangle
    filter: sed -e "s~ATOL~RTOL~g" -e "s~ABS~RELATIVE~g"
    args: -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ts_type beuler -ts_max_steps 10 -ts_dt 0.1 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1.0e-10 -fieldsplit_pressure_pc_type jacobi -ksp_monitor_short -ksp_converged_reason -snes_monitor_short -snes_converged_reason -ts_monitor
  test:
    suffix: 2d_q2q1_r1
    requires: !single
    filter: sed -e "s~ATOL~RTOL~g" -e "s~ABS~RELATIVE~g" -e "s~ 0\]~ 0.0\]~g"
    args: -simplex 0 -dm_refine 1 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ts_type beuler -ts_max_steps 10 -ts_dt 0.1 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1.0e-10 -fieldsplit_pressure_pc_type jacobi -ksp_monitor_short -ksp_converged_reason -snes_monitor_short -snes_converged_reason -ts_monitor
  test:
    suffix: 2d_q2q1_r2
    requires: !single
    filter: sed -e "s~ATOL~RTOL~g" -e "s~ABS~RELATIVE~g"
    args: -simplex 0 -dm_refine 2 -vel_petscspace_degree 2 -pres_petscspace_degree 1 -ts_type beuler -ts_max_steps 10 -ts_dt 0.1 -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -fieldsplit_velocity_pc_type lu -fieldsplit_pressure_ksp_rtol 1.0e-10 -fieldsplit_pressure_pc_type jacobi -ksp_monitor_short -ksp_converged_reason -snes_monitor_short -snes_converged_reason -ts_monitor

TEST*/
