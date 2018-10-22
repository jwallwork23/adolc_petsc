
static char help[] = "Power grid stability analysis of WECC 9 bus system.\n\
This example is based on the 9-bus (node) example given in the book Power\n\
Systems Dynamics and Stability (Chapter 7) by P. Sauer and M. A. Pai.\n\
The power grid in this example consists of 9 buses (nodes), 3 generators,\n\
3 loads, and 9 transmission lines. The network equations are written\n\
in current balance form using rectangular coordiantes.\n\n";

/*
   The equations for the stability analysis are described by the DAE

   \dot{x} = f(x,y,t)
     0     = g(x,y,t)

   where the generators are described by differential equations, while the algebraic
   constraints define the network equations.

   The generators are modeled with a 4th order differential equation describing the electrical
   and mechanical dynamics. Each generator also has an exciter system modeled by 3rd order
   diff. eqns. describing the exciter, voltage regulator, and the feedback stabilizer
   mechanism.

   The network equations are described by nodal current balance equations.
    I(x,y) - Y*V = 0

   where:
    I(x,y) is the current injected from generators and loads.
      Y    is the admittance matrix, and
      V    is the voltage vector
*/

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>
#include <adolc/adolc.h>
#include "utils/monitor.cpp"
#include "utils/jacobian.cpp"


int main(int argc,char **argv)
{
  TS             ts;
  SNES           snes_alg;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Userctx        user;
  AdolcCtx       *adctx;
  PetscViewer    Xview,Ybusview,viewer;
  Vec            X,F_alg,R;
  Mat            J,A;
  PetscInt       i,idx,*idx2;
  Vec            Xdot;
  PetscScalar    *x,*mat,*amat;
  Vec            vatol;
  PetscInt       *direction;
  PetscBool      *terminate,byhand = PETSC_FALSE;
  const PetscInt *idx3;
  PetscScalar    *vatoli;
  PetscInt       k;
  adouble        *xgen_a = NULL,*xnet_a = NULL,*fgen_a = NULL,*fnet_a = NULL,*xdot_a = NULL;


  ierr = PetscInitialize(&argc,&argv,"petscoptions",help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");
  ierr = PetscNew(&adctx);CHKERRQ(ierr);

  user.neqs_gen   = 9*ngen; /* # eqs. for generator subsystem */
  user.neqs_net   = 2*nbus; /* # eqs. for network subsystem   */
  user.neqs_pgrid = user.neqs_gen + user.neqs_net;
  user.adctx = adctx;
  adctx->m = user.neqs_pgrid;
  adctx->n = user.neqs_pgrid;
  adctx->p = user.neqs_pgrid;

  /* Create indices for differential and algebraic equations */

  ierr = PetscMalloc1(7*ngen,&idx2);CHKERRQ(ierr);
  for (i=0; i<ngen; i++) {
    idx2[7*i]   = 9*i;   idx2[7*i+1] = 9*i+1; idx2[7*i+2] = 9*i+2; idx2[7*i+3] = 9*i+3;
    idx2[7*i+4] = 9*i+6; idx2[7*i+5] = 9*i+7; idx2[7*i+6] = 9*i+8;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,7*ngen,idx2,PETSC_COPY_VALUES,&user.is_diff);CHKERRQ(ierr);
  ierr = ISComplement(user.is_diff,0,user.neqs_pgrid,&user.is_alg);CHKERRQ(ierr);
  ierr = PetscFree(idx2);CHKERRQ(ierr);

  /* Read initial voltage vector and Ybus */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"X.bin",FILE_MODE_READ,&Xview);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Ybus.bin",FILE_MODE_READ,&Ybusview);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&user.V0);CHKERRQ(ierr);
  ierr = VecSetSizes(user.V0,PETSC_DECIDE,user.neqs_net);CHKERRQ(ierr);
  ierr = VecLoad(user.V0,Xview);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user.Ybus);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Ybus,PETSC_DECIDE,PETSC_DECIDE,user.neqs_net,user.neqs_net);CHKERRQ(ierr);
  ierr = MatSetType(user.Ybus,MATBAIJ);CHKERRQ(ierr);
  /*  ierr = MatSetBlockSize(user.Ybus,2);CHKERRQ(ierr); */
  ierr = MatLoad(user.Ybus,Ybusview);CHKERRQ(ierr);

  /* Set run time options */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Transient stability fault options","");CHKERRQ(ierr);
  {
    user.tfaulton  = 1.0;
    user.tfaultoff = 1.2;
    user.Rfault    = 0.0001;
    user.setisdiff = PETSC_FALSE;
    user.semiexplicit = PETSC_FALSE;
    user.faultbus  = 8;
    ierr           = PetscOptionsReal("-tfaulton","","",user.tfaulton,&user.tfaulton,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsReal("-tfaultoff","","",user.tfaultoff,&user.tfaultoff,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsInt("-faultbus","","",user.faultbus,&user.faultbus,NULL);CHKERRQ(ierr);
    user.t0        = 0.0;
    user.tmax      = 5.0;
    ierr           = PetscOptionsReal("-t0","","",user.t0,&user.t0,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsReal("-tmax","","",user.tmax,&user.tmax,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsBool("-setisdiff","","",user.setisdiff,&user.setisdiff,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsBool("-dae_semiexplicit","","",user.semiexplicit,&user.semiexplicit,NULL);CHKERRQ(ierr);

    /* ADOL-C options */
    user.no_an     = PETSC_FALSE;
    ierr           = PetscOptionsBool("-no_annotation","","",user.no_an,&user.no_an,NULL);CHKERRQ(ierr);
    ierr           = PetscOptionsBool("-jacobian_by_hand","","",byhand,&byhand,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&Xview);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&Ybusview);CHKERRQ(ierr);

  /* Create DMs for generator and network subsystems */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,user.neqs_gen,1,1,NULL,&user.dmgen);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.dmgen,"dmgen_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.dmgen);CHKERRQ(ierr);
  ierr = DMSetUp(user.dmgen);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,user.neqs_net,1,1,NULL,&user.dmnet);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.dmnet,"dmnet_");CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.dmnet);CHKERRQ(ierr);
  ierr = DMSetUp(user.dmnet);CHKERRQ(ierr);
  /* Create a composite DM packer and add the two DMs */
  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&user.dmpgrid);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(user.dmpgrid,"pgrid_");CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.dmpgrid,user.dmgen);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.dmpgrid,user.dmnet);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.dmpgrid,&X);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,user.neqs_pgrid,user.neqs_pgrid);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = PreallocateJacobian(J,&user);CHKERRQ(ierr); // TODO: What about ADOL-C case?

  /* Create matrix to save solutions at each time step */
  user.stepnum = 0;

  ierr = MatCreateSeqDense(PETSC_COMM_SELF,user.neqs_pgrid+1,1002,NULL,&user.Sol);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SetInitialGuess(X,&user);CHKERRQ(ierr);

  ierr = VecDuplicate(X,&Xdot);CHKERRQ(ierr);
  if (!user.no_an) {

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Allocate memory for active variables
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    xgen_a = new adouble[user.neqs_gen];
    xnet_a = new adouble[user.neqs_net];
    fgen_a = new adouble[user.neqs_gen];
    fnet_a = new adouble[user.neqs_net];
    xdot_a = new adouble[user.neqs_pgrid];
    user.xgen_a = xgen_a;user.xnet_a = xnet_a;
    user.fgen_a = fgen_a;user.fnet_a = fnet_a;
    user.xdot_a = xdot_a;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Trace just once
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = VecDuplicate(X,&R);CHKERRQ(ierr);
    if (user.semiexplicit) {
      ierr = RHSFunctionActive(ts,0.,X,R,&user);CHKERRQ(ierr);
    } else {
      ierr = IFunctionActive(ts,0.,X,Xdot,R,&user);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&R);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  if (user.semiexplicit) {
    ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ts,NULL,RHSFunctionPassive,&user);CHKERRQ(ierr);
    if (byhand) {
      ierr = TSSetRHSJacobian(ts,J,J,RHSJacobianByHand,&user);CHKERRQ(ierr);
    } else {
      ierr = TSSetRHSJacobian(ts,J,J,RHSJacobianAdolc,&user);CHKERRQ(ierr);
    }
  } else {
    ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
    ierr = TSSetEquationType(ts,TS_EQ_DAE_IMPLICIT_INDEX1);CHKERRQ(ierr);
    ierr = TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunctionPassive,&user);CHKERRQ(ierr);
    if (byhand) {
      ierr = TSSetIJacobian(ts,J,J,(TSIJacobian)IJacobianByHand,&user);CHKERRQ(ierr);
    } else {
      ierr = TSSetIJacobian(ts,J,J,(TSIJacobian)IJacobianAdolc,&user);CHKERRQ(ierr);
    }
  }
  ierr = TSSetApplicationContext(ts,&user);CHKERRQ(ierr);

  /* Just to set up the Jacobian structure */
  if (byhand) {
    ierr = IJacobianByHand(ts,0.0,X,Xdot,0.0,J,J,&user);CHKERRQ(ierr);
  } else {
    if (user.semiexplicit) {
     ierr = RHSJacobianAdolc(ts,0.0,X,J,J,&user);CHKERRQ(ierr);
    } else {
     ierr = IJacobianAdolc(ts,0.0,X,Xdot,0.0,J,J,&user);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&Xdot);CHKERRQ(ierr);

  ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);  // TODO: temp

  /* Save initial solution */

  idx=user.stepnum*(user.neqs_pgrid+1);
  ierr = MatDenseGetArray(user.Sol,&mat);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);

  mat[idx] = 0.0;

  ierr = PetscMemcpy(mat+idx+1,x,user.neqs_pgrid*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(user.Sol,&mat);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  user.stepnum++;

  ierr = TSSetMaxTime(ts,user.tmax);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.01);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,SaveSolution);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);

  ierr = PetscMalloc1((2*ngen+2),&direction);CHKERRQ(ierr);
  ierr = PetscMalloc1((2*ngen+2),&terminate);CHKERRQ(ierr);
  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;
  for (i=0; i < ngen;i++) {
    direction[2+2*i] = -1; direction[2+2*i+1] = 1;
    terminate[2+2*i] = terminate[2+2*i+1] = PETSC_FALSE;
  }

  ierr = TSSetEventHandler(ts,2*ngen+2,direction,terminate,EventFunction,PostEventFunction,(void*)&user);CHKERRQ(ierr);

  if(user.semiexplicit) {
    /* Use a semi-explicit approach with the time-stepping done by an explicit method and the
       algrebraic part solved via PostStage and PostEvaluate callbacks 
    */
    ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
    ierr = TSSetPostStage(ts,PostStage);CHKERRQ(ierr);
    ierr = TSSetPostEvaluate(ts,PostEvaluate);CHKERRQ(ierr);
  }


  if(user.setisdiff) {
    /* Create vector of absolute tolerances and set the algebraic part to infinity */
    ierr = VecDuplicate(X,&vatol);CHKERRQ(ierr);
    ierr = VecSet(vatol,100000.0);CHKERRQ(ierr);
    ierr = VecGetArray(vatol,&vatoli);CHKERRQ(ierr);
    ierr = ISGetIndices(user.is_diff,&idx3);CHKERRQ(ierr);
    for(k=0; k < 7*ngen; k++) vatoli[idx3[k]] = 1e-2;
    ierr = VecRestoreArray(vatol,&vatoli);CHKERRQ(ierr);
  }

  /* Create the nonlinear solver for solving the algebraic system */
  /* Note that although the algebraic system needs to be solved only for
     Idq and V, we reuse the entire system including xgen. The xgen
     variables are held constant by setting their residuals to 0 and
     putting a 1 on the Jacobian diagonal for xgen rows
  */

  ierr = VecDuplicate(X,&F_alg);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes_alg);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes_alg,F_alg,AlgFunction,&user);CHKERRQ(ierr);     // TODO
  ierr = SNESSetJacobian(snes_alg,J,J,AlgJacobianByHand,&user);CHKERRQ(ierr); // TODO

  ierr = SNESSetFromOptions(snes_alg);CHKERRQ(ierr);

  user.snes_alg=snes_alg;
  
  /* Solve */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(user.Sol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user.Sol,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateSeqDense(PETSC_COMM_SELF,user.neqs_pgrid+1,user.stepnum,NULL,&A);CHKERRQ(ierr);
  ierr = MatDenseGetArray(user.Sol,&mat);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&amat);CHKERRQ(ierr);
  ierr = PetscMemcpy(amat,mat,(user.stepnum*(user.neqs_pgrid+1))*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(A,&amat);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(user.Sol,&mat);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"out.bin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space and call destructors for AFields.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFree(direction);CHKERRQ(ierr);
  ierr = PetscFree(terminate);CHKERRQ(ierr);
  if (!user.no_an) {
    delete[] xdot_a;
    delete[] fnet_a;
    delete[] fgen_a;
    delete[] xnet_a;
    delete[] xgen_a;
  }
  ierr = SNESDestroy(&snes_alg);CHKERRQ(ierr);
  ierr = VecDestroy(&F_alg);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Ybus);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Sol);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&user.V0);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dmgen);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dmnet);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dmpgrid);CHKERRQ(ierr);
  ierr = ISDestroy(&user.is_diff);CHKERRQ(ierr);
  ierr = ISDestroy(&user.is_alg);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if(user.setisdiff) {
    ierr = VecDestroy(&vatol);CHKERRQ(ierr);
  }
  ierr = PetscFree(adctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: double !complex !define(PETSC_USE_64BIT_INDICES)

   test:
      suffix: implicit
      args: -ts_monitor -snes_monitor_short
      localrunfiles: petscoptions X.bin Ybus.bin

   test:
      suffix: semiexplicit
      args: -ts_monitor -snes_monitor_short -dae_semiexplicit -ts_rk_type 2a
      localrunfiles: petscoptions X.bin Ybus.bin

   test:
      suffix: steprestart
      args: -ts_monitor -snes_monitor_short -ts_type arkimex
      localrunfiles: petscoptions X.bin Ybus.bin

TEST*/
