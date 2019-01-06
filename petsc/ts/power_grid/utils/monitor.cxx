#include "init.cxx"

/*
  The first two events are for fault on and off, respectively. The following events are
  to check the min/max limits on the state variable VR. A non windup limiter is used for
  the VR limits.
*/
PetscErrorCode EventFunction(TS ts,PetscReal t,Vec X,PetscScalar *fvalue,void *ctx)
{
  Userctx        *user=(Userctx*)ctx;
  Vec            Xgen,Xnet;
  PetscInt       i,idx=0;
  const PetscScalar *xgen,*xnet;
  PetscErrorCode ierr;
  PetscScalar    Efd,RF,VR,Vr,Vi,Vm;

  PetscFunctionBegin;

  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->dmpgrid,X,Xgen,Xnet);CHKERRQ(ierr);

  ierr = VecGetArrayRead(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xnet,&xnet);CHKERRQ(ierr);

  /* Event for fault-on time */
  fvalue[0] = t - user->tfaulton;
  /* Event for fault-off time */
  fvalue[1] = t - user->tfaultoff;

  for (i=0; i < ngen; i++) {
    Efd   = xgen[idx+6];
    RF    = xgen[idx+7];
    VR    = xgen[idx+8];

    Vr = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
    Vi = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */
    Vm = PetscSqrtScalar(Vr*Vr + Vi*Vi);

    if (!VRatmax[i]) {
      fvalue[2+2*i] = VRMAX[i] - VR;
    } else {
      fvalue[2+2*i] = (VR - KA[i]*RF + KA[i]*KF[i]*Efd/TF[i] - KA[i]*(Vref[i] - Vm))/TA[i];
    }
    if (!VRatmin[i]) {
      fvalue[2+2*i+1] = VRMIN[i] - VR;
    } else {
      fvalue[2+2*i+1] = (VR - KA[i]*RF + KA[i]*KF[i]*Efd/TF[i] - KA[i]*(Vref[i] - Vm))/TA[i];
    }
    idx = idx+9;
  }
  ierr = VecRestoreArrayRead(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xnet,&xnet);CHKERRQ(ierr);

  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec X,PetscBool forwardsolve,void* ctx)
{
  Userctx *user=(Userctx*)ctx;
  Vec      Xgen,Xnet;
  PetscScalar *xgen,*xnet;
  PetscInt row_loc,col_loc;
  PetscScalar val;
  PetscErrorCode ierr;
  PetscInt i,idx=0,event_num;
  PetscScalar fvalue;
  PetscScalar Efd, RF, VR;
  PetscScalar Vr,Vi,Vm;

  PetscFunctionBegin;

  ierr = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->dmpgrid,X,Xgen,Xnet);CHKERRQ(ierr);

  ierr = VecGetArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecGetArray(Xnet,&xnet);CHKERRQ(ierr);

  for (i=0; i < nevents; i++) {
    if (event_list[i] == 0) {
      /* Apply disturbance - resistive fault at user->faultbus */
      /* This is done by adding shunt conductance to the diagonal location
         in the Ybus matrix */
      row_loc = 2*user->faultbus; col_loc = 2*user->faultbus+1; /* Location for G */
      val     = 1/user->Rfault;
      ierr    = MatSetValues(user->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);
      row_loc = 2*user->faultbus+1; col_loc = 2*user->faultbus; /* Location for G */
      val     = 1/user->Rfault;
      ierr    = MatSetValues(user->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);

      ierr = MatAssemblyBegin(user->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(user->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      /* Solve the algebraic equations */
      ierr = SNESSolve(user->snes_alg,NULL,X);CHKERRQ(ierr);
    } else if(event_list[i] == 1) {
      /* Remove the fault */
      row_loc = 2*user->faultbus; col_loc = 2*user->faultbus+1;
      val     = -1/user->Rfault;
      ierr    = MatSetValues(user->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);
      row_loc = 2*user->faultbus+1; col_loc = 2*user->faultbus;
      val     = -1/user->Rfault;
      ierr    = MatSetValues(user->Ybus,1,&row_loc,1,&col_loc,&val,ADD_VALUES);CHKERRQ(ierr);

      ierr = MatAssemblyBegin(user->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(user->Ybus,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

      /* Solve the algebraic equations */
      ierr = SNESSolve(user->snes_alg,NULL,X);CHKERRQ(ierr);

      /* Check the VR derivatives and reset flags if needed */
      for (i=0; i < ngen; i++) {
        Efd   = xgen[idx+6];
        RF    = xgen[idx+7];
        VR    = xgen[idx+8];

        Vr = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
        Vi = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */
        Vm = PetscSqrtScalar(Vr*Vr + Vi*Vi);

        if (VRatmax[i]) {
          fvalue = (VR - KA[i]*RF + KA[i]*KF[i]*Efd/TF[i] - KA[i]*(Vref[i] - Vm))/TA[i];
          if (fvalue < 0) {
            VRatmax[i] = 0;
            ierr = PetscPrintf(PETSC_COMM_SELF,"VR[%d]: dVR_dt went negative on fault clearing at time %g\n",i,t);CHKERRQ(ierr);
          }
        }
        if (VRatmin[i]) {
          fvalue =  (VR - KA[i]*RF + KA[i]*KF[i]*Efd/TF[i] - KA[i]*(Vref[i] - Vm))/TA[i];

          if(fvalue > 0) {
            VRatmin[i] = 0;
            ierr = PetscPrintf(PETSC_COMM_SELF,"VR[%d]: dVR_dt went positive on fault clearing at time %g\n",i,t);CHKERRQ(ierr);
          }
        }
        idx = idx+9;
      }
    } else {
      idx = (event_list[i]-2)/2;
      event_num = (event_list[i]-2)%2;
      if (event_num == 0) { /* Max VR */
        if (!VRatmax[idx]) {
          VRatmax[idx] = 1;
          ierr = PetscPrintf(PETSC_COMM_SELF,"VR[%d]: hit upper limit at time %g\n",idx,t);CHKERRQ(ierr);
        }
        else {
          VRatmax[idx] = 0;
          ierr = PetscPrintf(PETSC_COMM_SELF,"VR[%d]: freeing variable as dVR_dt is negative at time %g\n",idx,t);CHKERRQ(ierr);
        }
      } else {
        if (!VRatmin[idx]) {
          VRatmin[idx] = 1;
          ierr = PetscPrintf(PETSC_COMM_SELF,"VR[%d]: hit lower limit at time %g\n",idx,t);CHKERRQ(ierr);
        }
        else {
          VRatmin[idx] = 0;
          ierr = PetscPrintf(PETSC_COMM_SELF,"VR[%d]: freeing variable as dVR_dt is positive at time %g\n",idx,t);CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = VecRestoreArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xnet,&xnet);CHKERRQ(ierr);

  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Saves the solution at each time to a matrix */
PetscErrorCode SaveSolution(TS ts)
{
  PetscErrorCode    ierr;
  Userctx           *user;
  Vec               X;
  const PetscScalar *x;
  PetscScalar       *mat;
  PetscInt          idx;
  PetscReal         t;

  PetscFunctionBegin;
  ierr     = TSGetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr     = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr     = TSGetSolution(ts,&X);CHKERRQ(ierr);
  idx      = user->stepnum*(user->neqs_pgrid+1);
  ierr     = MatDenseGetArray(user->Sol,&mat);CHKERRQ(ierr);
  ierr     = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  mat[idx] = t;
  ierr     = PetscMemcpy(mat+idx+1,x,user->neqs_pgrid*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr     = MatDenseRestoreArray(user->Sol,&mat);CHKERRQ(ierr);
  ierr     = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  user->stepnum++;
  PetscFunctionReturn(0);
}

PetscErrorCode PostStage(TS ts, PetscReal t, PetscInt i, Vec *X)
{
  PetscErrorCode ierr;
  Userctx        *user;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr = SNESSolve(user->snes_alg,NULL,X[i]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PostEvaluate(TS ts)
{
  PetscErrorCode ierr;
  Userctx        *user;
  Vec            X;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr = TSGetSolution(ts,&X);CHKERRQ(ierr);
  ierr = SNESSolve(user->snes_alg,NULL,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

