#include "tracing.cpp"
#include "../../../utils/drivers.cpp"


PetscErrorCode PreallocateJacobian(Mat J, Userctx *user)
{
  PetscErrorCode ierr;
  PetscInt       *d_nnz;
  PetscInt       i,idx=0,start=0;
  PetscInt       ncols;

  PetscFunctionBegin;
  ierr = PetscMalloc1(user->neqs_pgrid,&d_nnz);CHKERRQ(ierr);
  for (i=0; i<user->neqs_pgrid; i++) d_nnz[i] = 0;
  /* Generator subsystem */
  for (i=0; i < ngen; i++) {

    d_nnz[idx]   += 3;
    d_nnz[idx+1] += 2;
    d_nnz[idx+2] += 2;
    d_nnz[idx+3] += 5;
    d_nnz[idx+4] += 6;
    d_nnz[idx+5] += 6;

    d_nnz[user->neqs_gen+2*gbus[i]]   += 3;
    d_nnz[user->neqs_gen+2*gbus[i]+1] += 3;

    d_nnz[idx+6] += 2;
    d_nnz[idx+7] += 2;
    d_nnz[idx+8] += 5;

    idx = idx + 9;
  }

  start = user->neqs_gen;

  for (i=0; i < nbus; i++) {
    ierr = MatGetRow(user->Ybus,2*i,&ncols,NULL,NULL);CHKERRQ(ierr);
    d_nnz[start+2*i]   += ncols;
    d_nnz[start+2*i+1] += ncols;
    ierr = MatRestoreRow(user->Ybus,2*i,&ncols,NULL,NULL);CHKERRQ(ierr);
  }

  ierr = MatSeqAIJSetPreallocation(J,0,d_nnz);CHKERRQ(ierr);

  ierr = PetscFree(d_nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   J = [df_dx, df_dy
        dg_dx, dg_dy]
*/
PetscErrorCode ResidualJacobianByHand(Vec X,Mat J,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  Userctx        *user=(Userctx*)ctx;
  Vec            Xgen,Xnet;
  PetscScalar    *xgen,*xnet;
  PetscInt       i,idx=0;
  PetscScalar    Vr,Vi,Vm,Vm2;
  PetscScalar    Eqp,Edp,delta; /* Generator variables */
  PetscScalar    Efd;
  PetscScalar    Id,Iq;  /* Generator dq axis currents */
  PetscScalar    Vd,Vq;
  PetscScalar    val[10];
  PetscInt       row[2],col[10];
  PetscInt       net_start=user->neqs_gen;
  PetscScalar    Zdq_inv[4],det;
  PetscScalar    dVd_dVr,dVd_dVi,dVq_dVr,dVq_dVi,dVd_ddelta,dVq_ddelta;
  PetscScalar    dIGr_ddelta,dIGi_ddelta,dIGr_dId,dIGr_dIq,dIGi_dId,dIGi_dIq;
  PetscScalar    dSE_dEfd;
  PetscScalar    dVm_dVd,dVm_dVq,dVm_dVr,dVm_dVi;
  PetscInt          ncols;
  const PetscInt    *cols;
  const PetscScalar *yvals;
  PetscInt          k;
  PetscScalar PD,QD,Vm0,*v0,Vm4;
  PetscScalar dPD_dVr,dPD_dVi,dQD_dVr,dQD_dVi;
  PetscScalar dIDr_dVr,dIDr_dVi,dIDi_dVr,dIDi_dVi;


  PetscFunctionBegin;
  ierr  = MatZeroEntries(B);CHKERRQ(ierr);
  ierr  = DMCompositeGetLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);
  ierr  = DMCompositeScatter(user->dmpgrid,X,Xgen,Xnet);CHKERRQ(ierr);

  ierr = VecGetArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecGetArray(Xnet,&xnet);CHKERRQ(ierr);

  /* Generator subsystem */
  for (i=0; i < ngen; i++) {
    Eqp   = xgen[idx];
    Edp   = xgen[idx+1];
    delta = xgen[idx+2];
    Id    = xgen[idx+4];
    Iq    = xgen[idx+5];
    Efd   = xgen[idx+6];

    /*    fgen[idx]   = (-Eqp - (Xd[i] - Xdp[i])*Id + Efd)/Td0p[i]; */
    row[0] = idx;
    col[0] = idx;           col[1] = idx+4;          col[2] = idx+6;
    val[0] = -1/ Td0p[i]; val[1] = -(Xd[i] - Xdp[i])/ Td0p[i]; val[2] = 1/Td0p[i];

    ierr = MatSetValues(J,1,row,3,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /*    fgen[idx+1] = (-Edp + (Xq[i] - Xqp[i])*Iq)/Tq0p[i]; */
    row[0] = idx + 1;
    col[0] = idx + 1;       col[1] = idx+5;
    val[0] = -1/Tq0p[i]; val[1] = (Xq[i] - Xqp[i])/Tq0p[i];
    ierr   = MatSetValues(J,1,row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /*    fgen[idx+2] = w - w_s; */
    row[0] = idx + 2;
    col[0] = idx + 2; col[1] = idx + 3;
    val[0] = 0;       val[1] = 1;
    ierr   = MatSetValues(J,1,row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /*    fgen[idx+3] = (TM[i] - Edp*Id - Eqp*Iq - (Xqp[i] - Xdp[i])*Id*Iq - D[i]*(w - w_s))/M[i]; */
    row[0] = idx + 3;
    col[0] = idx; col[1] = idx + 1; col[2] = idx + 3;       col[3] = idx + 4;                  col[4] = idx + 5;
    val[0] = -Iq/M[i];  val[1] = -Id/M[i];      val[2] = -D[i]/M[i]; val[3] = (-Edp - (Xqp[i]-Xdp[i])*Iq)/M[i]; val[4] = (-Eqp - (Xqp[i] - Xdp[i])*Id)/M[i];
    ierr   = MatSetValues(J,1,row,5,col,val,INSERT_VALUES);CHKERRQ(ierr);

    Vr   = xnet[2*gbus[i]]; /* Real part of generator terminal voltage */
    Vi   = xnet[2*gbus[i]+1]; /* Imaginary part of the generator terminal voltage */
    ierr = ri2dq(Vr,Vi,delta,&Vd,&Vq);CHKERRQ(ierr);

    det = Rs[i]*Rs[i] + Xdp[i]*Xqp[i];

    Zdq_inv[0] = Rs[i]/det;
    Zdq_inv[1] = Xqp[i]/det;
    Zdq_inv[2] = -Xdp[i]/det;
    Zdq_inv[3] = Rs[i]/det;

    dVd_dVr    = PetscSinScalar(delta); dVd_dVi = -PetscCosScalar(delta);
    dVq_dVr    = PetscCosScalar(delta); dVq_dVi = PetscSinScalar(delta);
    dVd_ddelta = Vr*PetscCosScalar(delta) + Vi*PetscSinScalar(delta);
    dVq_ddelta = -Vr*PetscSinScalar(delta) + Vi*PetscCosScalar(delta);

    /*    fgen[idx+4] = Zdq_inv[0]*(-Edp + Vd) + Zdq_inv[1]*(-Eqp + Vq) + Id; */
    row[0] = idx+4;
    col[0] = idx;         col[1] = idx+1;        col[2] = idx + 2;
    val[0] = -Zdq_inv[1]; val[1] = -Zdq_inv[0];  val[2] = Zdq_inv[0]*dVd_ddelta + Zdq_inv[1]*dVq_ddelta;
    col[3] = idx + 4; col[4] = net_start+2*gbus[i];                     col[5] = net_start + 2*gbus[i]+1;
    val[3] = 1;       val[4] = Zdq_inv[0]*dVd_dVr + Zdq_inv[1]*dVq_dVr; val[5] = Zdq_inv[0]*dVd_dVi + Zdq_inv[1]*dVq_dVi;
    ierr   = MatSetValues(J,1,row,6,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /*  fgen[idx+5] = Zdq_inv[2]*(-Edp + Vd) + Zdq_inv[3]*(-Eqp + Vq) + Iq; */
    row[0] = idx+5;
    col[0] = idx;         col[1] = idx+1;        col[2] = idx + 2;
    val[0] = -Zdq_inv[3]; val[1] = -Zdq_inv[2];  val[2] = Zdq_inv[2]*dVd_ddelta + Zdq_inv[3]*dVq_ddelta;
    col[3] = idx + 5; col[4] = net_start+2*gbus[i];                     col[5] = net_start + 2*gbus[i]+1;
    val[3] = 1;       val[4] = Zdq_inv[2]*dVd_dVr + Zdq_inv[3]*dVq_dVr; val[5] = Zdq_inv[2]*dVd_dVi + Zdq_inv[3]*dVq_dVi;
    ierr   = MatSetValues(J,1,row,6,col,val,INSERT_VALUES);CHKERRQ(ierr);

    dIGr_ddelta = Id*PetscCosScalar(delta) - Iq*PetscSinScalar(delta);
    dIGi_ddelta = Id*PetscSinScalar(delta) + Iq*PetscCosScalar(delta);
    dIGr_dId    = PetscSinScalar(delta);  dIGr_dIq = PetscCosScalar(delta);
    dIGi_dId    = -PetscCosScalar(delta); dIGi_dIq = PetscSinScalar(delta);

    /* fnet[2*gbus[i]]   -= IGi; */
    row[0] = net_start + 2*gbus[i];
    col[0] = idx+2;        col[1] = idx + 4;   col[2] = idx + 5;
    val[0] = -dIGi_ddelta; val[1] = -dIGi_dId; val[2] = -dIGi_dIq;
    ierr = MatSetValues(J,1,row,3,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /* fnet[2*gbus[i]+1]   -= IGr; */
    row[0] = net_start + 2*gbus[i]+1;
    col[0] = idx+2;        col[1] = idx + 4;   col[2] = idx + 5;
    val[0] = -dIGr_ddelta; val[1] = -dIGr_dId; val[2] = -dIGr_dIq;
    ierr   = MatSetValues(J,1,row,3,col,val,INSERT_VALUES);CHKERRQ(ierr);

    Vm = PetscSqrtScalar(Vd*Vd + Vq*Vq);

    /*    fgen[idx+6] = (-KE[i]*Efd - SE + VR)/TE[i]; */
    /*    SE  = k1[i]*PetscExpScalar(k2[i]*Efd); */

    dSE_dEfd = k1[i]*k2[i]*PetscExpScalar(k2[i]*Efd);

    row[0] = idx + 6;
    col[0] = idx + 6;                     col[1] = idx + 8;
    val[0] = (-KE[i] - dSE_dEfd)/TE[i];  val[1] = 1/TE[i];
    ierr   = MatSetValues(J,1,row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /* Exciter differential equations */

    /*    fgen[idx+7] = (-RF + KF[i]*Efd/TF[i])/TF[i]; */
    row[0] = idx + 7;
    col[0] = idx + 6;       col[1] = idx + 7;
    val[0] = (KF[i]/TF[i])/TF[i];  val[1] = -1/TF[i];
    ierr   = MatSetValues(J,1,row,2,col,val,INSERT_VALUES);CHKERRQ(ierr);

    /*    fgen[idx+8] = (-VR + KA[i]*RF - KA[i]*KF[i]*Efd/TF[i] + KA[i]*(Vref[i] - Vm))/TA[i]; */
    /* Vm = (Vd^2 + Vq^2)^0.5; */

    row[0] = idx + 8;
    if(VRatmax[i]) {
      col[0] = idx + 8; val[0] = 1.0;
      ierr = MatSetValues(J,1,row,1,col,val,INSERT_VALUES);CHKERRQ(ierr);
    } else if(VRatmin[i]) {
      col[0] = idx + 8; val[0] = -1.0;
      ierr = MatSetValues(J,1,row,1,col,val,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      dVm_dVd    = Vd/Vm; dVm_dVq = Vq/Vm;
      dVm_dVr    = dVm_dVd*dVd_dVr + dVm_dVq*dVq_dVr;
      dVm_dVi    = dVm_dVd*dVd_dVi + dVm_dVq*dVq_dVi;
      row[0]     = idx + 8;
      col[0]     = idx + 6;           col[1] = idx + 7; col[2] = idx + 8;
      val[0]     = -(KA[i]*KF[i]/TF[i])/TA[i]; val[1] = KA[i]/TA[i];  val[2] = -1/TA[i];
      col[3]     = net_start + 2*gbus[i]; col[4] = net_start + 2*gbus[i]+1;
      val[3]     = -KA[i]*dVm_dVr/TA[i];         val[4] = -KA[i]*dVm_dVi/TA[i];
      ierr       = MatSetValues(J,1,row,5,col,val,INSERT_VALUES);CHKERRQ(ierr);
    }

    idx        = idx + 9;
  }

  for (i=0; i<nbus; i++) {
    ierr   = MatGetRow(user->Ybus,2*i,&ncols,&cols,&yvals);CHKERRQ(ierr);
    row[0] = net_start + 2*i;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    ierr = MatSetValues(J,1,row,ncols,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(user->Ybus,2*i,&ncols,&cols,&yvals);CHKERRQ(ierr);

    ierr   = MatGetRow(user->Ybus,2*i+1,&ncols,&cols,&yvals);CHKERRQ(ierr);
    row[0] = net_start + 2*i+1;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    ierr = MatSetValues(J,1,row,ncols,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(user->Ybus,2*i+1,&ncols,&cols,&yvals);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(J,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FLUSH_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecGetArray(user->V0,&v0);CHKERRQ(ierr);
  for (i=0; i < nload; i++) {
    Vr      = xnet[2*lbus[i]]; /* Real part of load bus voltage */
    Vi      = xnet[2*lbus[i]+1]; /* Imaginary part of the load bus voltage */
    Vm      = PetscSqrtScalar(Vr*Vr + Vi*Vi); Vm2 = Vm*Vm; Vm4 = Vm2*Vm2;
    Vm0     = PetscSqrtScalar(v0[2*lbus[i]]*v0[2*lbus[i]] + v0[2*lbus[i]+1]*v0[2*lbus[i]+1]);
    PD      = QD = 0.0;
    dPD_dVr = dPD_dVi = dQD_dVr = dQD_dVi = 0.0;
    for (k=0; k < ld_nsegsp[i]; k++) {
      PD      += ld_alphap[k]*PD0[i]*PetscPowScalar((Vm/Vm0),ld_betap[k]);
      dPD_dVr += ld_alphap[k]*ld_betap[k]*PD0[i]*PetscPowScalar((1/Vm0),ld_betap[k])*Vr*PetscPowScalar(Vm,(ld_betap[k]-2));
      dPD_dVi += ld_alphap[k]*ld_betap[k]*PD0[i]*PetscPowScalar((1/Vm0),ld_betap[k])*Vi*PetscPowScalar(Vm,(ld_betap[k]-2));
    }
    for (k=0; k < ld_nsegsq[i]; k++) {
      QD      += ld_alphaq[k]*QD0[i]*PetscPowScalar((Vm/Vm0),ld_betaq[k]);
      dQD_dVr += ld_alphaq[k]*ld_betaq[k]*QD0[i]*PetscPowScalar((1/Vm0),ld_betaq[k])*Vr*PetscPowScalar(Vm,(ld_betaq[k]-2));
      dQD_dVi += ld_alphaq[k]*ld_betaq[k]*QD0[i]*PetscPowScalar((1/Vm0),ld_betaq[k])*Vi*PetscPowScalar(Vm,(ld_betaq[k]-2));
    }

    /*    IDr = (PD*Vr + QD*Vi)/Vm2; */
    /*    IDi = (-QD*Vr + PD*Vi)/Vm2; */

    dIDr_dVr = (dPD_dVr*Vr + dQD_dVr*Vi + PD)/Vm2 - ((PD*Vr + QD*Vi)*2*Vr)/Vm4;
    dIDr_dVi = (dPD_dVi*Vr + dQD_dVi*Vi + QD)/Vm2 - ((PD*Vr + QD*Vi)*2*Vi)/Vm4;

    dIDi_dVr = (-dQD_dVr*Vr + dPD_dVr*Vi - QD)/Vm2 - ((-QD*Vr + PD*Vi)*2*Vr)/Vm4;
    dIDi_dVi = (-dQD_dVi*Vr + dPD_dVi*Vi + PD)/Vm2 - ((-QD*Vr + PD*Vi)*2*Vi)/Vm4;


    /*    fnet[2*lbus[i]]   += IDi; */
    row[0] = net_start + 2*lbus[i];
    col[0] = net_start + 2*lbus[i];  col[1] = net_start + 2*lbus[i]+1;
    val[0] = dIDi_dVr;               val[1] = dIDi_dVi;
    ierr   = MatSetValues(J,1,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
    /*    fnet[2*lbus[i]+1] += IDr; */
    row[0] = net_start + 2*lbus[i]+1;
    col[0] = net_start + 2*lbus[i];  col[1] = net_start + 2*lbus[i]+1;
    val[0] = dIDr_dVr;               val[1] = dIDr_dVi;
    ierr   = MatSetValues(J,1,row,2,col,val,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(user->V0,&v0);CHKERRQ(ierr);

  ierr = VecRestoreArray(Xgen,&xgen);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xnet,&xnet);CHKERRQ(ierr);

  ierr = DMCompositeRestoreLocalVectors(user->dmpgrid,&Xgen,&Xnet);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   J = [I, 0
        dg_dx, dg_dy]
*/
PetscErrorCode AlgJacobianByHand(SNES snes,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  Userctx        *user=(Userctx*)ctx;

  PetscFunctionBegin;
  ierr = ResidualJacobianByHand(X,A,B,ctx);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatZeroRowsIS(A,user->is_diff,1.0,NULL,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   J = [-df_dx, -df_dy
        dg_dx, dg_dy]
*/

PetscErrorCode RHSJacobianByHand(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  Userctx        *user=(Userctx*)ctx;

  PetscFunctionBegin;
  user->t = t;
  ierr = ResidualJacobianByHand(X,A,B,user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   J = [df_dx-aI, df_dy
        dg_dx, dg_dy]
*/

PetscErrorCode IJacobianByHand(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,Userctx *user)
{
  PetscErrorCode ierr;
  PetscScalar    atmp = (PetscScalar) a;
  PetscInt       i,row;

  PetscFunctionBegin;
  user->t = t;
  atmp *= -1;

  ierr = RHSJacobianByHand(ts,t,X,A,B,user);CHKERRQ(ierr);
  for (i=0;i < ngen;i++) {
    row = 9*i;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+1;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+2;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+3;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+6;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+7;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
    row  = 9*i+8;
    ierr = MatSetValues(A,1,&row,1,&row,&atmp,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------  ADOL-C versions -------------------------------- */

PetscErrorCode ResidualJacobianAdolc(Vec X,Mat J,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  Userctx        *user = (Userctx*)ctx;
  PetscScalar    *x_vec;
  PetscScalar    val[10];
  PetscInt       i,k,ncols,row[2],col[10];
  PetscInt       net_start=user->neqs_gen;
  const PetscInt    *cols;
  const PetscScalar *yvals;

  PetscFunctionBegin;
  ierr = MatSetOption(J,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x_vec);CHKERRQ(ierr);
  ierr = AdolcComputeRHSJacobian(J,x_vec,user->adctx);CHKERRQ(ierr);

  // Manual differentiation of MatMult:

  for (i=0; i<nbus; i++) {
    ierr   = MatGetRow(user->Ybus,2*i,&ncols,&cols,&yvals);CHKERRQ(ierr);
    row[0] = net_start + 2*i;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    ierr = MatSetValues(J,1,row,ncols,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(user->Ybus,2*i,&ncols,&cols,&yvals);CHKERRQ(ierr);

    ierr   = MatGetRow(user->Ybus,2*i+1,&ncols,&cols,&yvals);CHKERRQ(ierr);
    row[0] = net_start + 2*i+1;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    ierr = MatSetValues(J,1,row,ncols,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(user->Ybus,2*i+1,&ncols,&cols,&yvals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArray(X,&x_vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobianAdolc(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  Userctx        *user=(Userctx*)ctx;

  PetscFunctionBegin;
  user->t = t;
  ierr = ResidualJacobianAdolc(X,A,B,user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode IJacobianAdolc(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,Userctx *user)
{
  PetscErrorCode ierr;
  PetscScalar    *x_vec;
  Vec            Xcopy;
  PetscScalar    val[10];
  PetscInt       i,k,ncols,row[2],col[10];
  PetscInt       net_start=user->neqs_gen;
  const PetscInt    *cols;
  const PetscScalar *yvals;

  PetscFunctionBegin;

  user->t = t;
  ierr = VecDuplicate(X,&Xcopy);CHKERRQ(ierr);  // X is read-only
  ierr = VecCopy(X,Xcopy);CHKERRQ(ierr);        // Copy values over
  ierr = VecGetArray(Xcopy,&x_vec);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr); // FIXME
  ierr = AdolcComputeIJacobian(A,x_vec,a,user->adctx);CHKERRQ(ierr);

  // Manual differentiation of MatMult:

  for (i=0; i<nbus; i++) {
    ierr   = MatGetRow(user->Ybus,2*i,&ncols,&cols,&yvals);CHKERRQ(ierr);
    row[0] = net_start + 2*i;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    ierr = MatSetValues(A,1,row,ncols,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(user->Ybus,2*i,&ncols,&cols,&yvals);CHKERRQ(ierr);

    ierr   = MatGetRow(user->Ybus,2*i+1,&ncols,&cols,&yvals);CHKERRQ(ierr);
    row[0] = net_start + 2*i+1;
    for (k=0; k<ncols; k++) {
      col[k] = net_start + cols[k];
      val[k] = yvals[k];
    }
    ierr = MatSetValues(A,1,row,ncols,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(user->Ybus,2*i+1,&ncols,&cols,&yvals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArray(Xcopy,&x_vec);CHKERRQ(ierr);
  ierr = VecDestroy(&Xcopy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
