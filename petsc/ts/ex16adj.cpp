static char help[] = "Performs adjoint sensitivity analysis for the van der Pol equation.\n\
Input parameters include:\n\
      -mu : stiffness parameter\n\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation
   Concepts: TS^adjoint sensitivity analysis
   Processors: 1
*/
/* ------------------------------------------------------------------------

   This program solves the van der Pol equation
       y'' - \mu (1-y^2)*y' + y = 0        (1)
   on the domain 0 <= x <= 1, with the boundary conditions
       y(0) = 2, y'(0) = 0,
   and computes the sensitivities of the final solution w.r.t. initial conditions and parameter \mu with an explicit Runge-Kutta method and its discrete adjoint.

   Notes:
   This code demonstrates the TSAdjoint interface to a system of ordinary differential equations (ODEs) in the form of u_t = F(u,t).

   (1) can be turned into a system of first order ODEs
   [ y' ] = [          z          ]
   [ z' ]   [ \mu (1 - y^2) z - y ]

   which then we can write as a vector equation

   [ u_1' ] = [             u_2           ]  (2)
   [ u_2' ]   [ \mu (1 - u_1^2) u_2 - u_1 ]

   which is now in the form of u_t = F(u,t).

   The user provides the right-hand-side function

   [ F(u,t) ] = [ u_2                       ]
                [ \mu (1 - u_1^2) u_2 - u_1 ]

   the Jacobian function

   dF   [       0           ;         1        ]
   -- = [                                      ]
   du   [ -2 \mu u_1*u_2 - 1;  \mu (1 - u_1^2) ]

   and the JacobianP (the Jacobian w.r.t. parameter) function

   dF      [       0          ]
   ---   = [                  ]
   d\mu    [ (1 - u_1^2) u_2  ]


  ------------------------------------------------------------------------- */

#include <petscts.h>
#include <petscmat.h>
#include <adolc/adolc.h>	/* ##### Include ADOL-C ##### */

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
  PetscReal tprev;
};

/*
*  User-defined routines
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscReal   mu   = user->mu;
  PetscScalar       *f;
  const PetscScalar *x;
 
  adouble           f_a[2];   	// ##### adouble for dependent variables #####
  adouble           x_a[2];   	// ##### adouble for independent variables #####

  adouble           fp_a[2];
  adouble           mu_a;	// ##### adouble for mu parameter #####

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);      // Get values for passive independent variables
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);          // Get array for passive dependent variables

  trace_on(1);			// ##### Start of active section for df/dx #####
  x_a[0] <<= x[0]; x_a[1] <<= x[1];	// Mark as independent
  f_a[0] = x_a[1];
  f_a[1] = mu*(1.-x_a[0]*x_a[0])*x_a[1]-x_a[0];
  f_a[0] >>= f[0]; f_a[1] >>= f[1];	// Mark as dependent
  trace_off(1);			// ##### End of active section for df/dx #####

  trace_on(2);			// ##### Start of active section for dmu/dx #####
  mu_a <<= mu;
  fp_a[0] = x[1];
  fp_a[1] = mu_a*(1.-x[0]*x[0])*x[1]-x[0];
  fp_a[0] >>= f[0]; fp_a[1] >>= f[1];
  trace_off(2);			// ##### End of active section for dmu/dx #####

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);  // Restore passive indep. variable array
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);      // Give values to passive dep. varible array
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscReal   mu   = user->mu;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  //PetscScalar       *Jx;			// TODO: use PetscMalloc1
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  // ##### Evaluate Jacobian using ADOL-C #####
  PetscScalar** Jx = (PetscScalar**) malloc(2*sizeof(PetscScalar*));
  Jx[0] = (PetscScalar*)malloc(2*sizeof(PetscScalar));
  Jx[1] = (PetscScalar*)malloc(2*sizeof(PetscScalar));
  
  //ierr = PetscMalloc1(4,&Jx);CHKERRQ(ierr);	// TODO: use PetscMalloc1
  //ierr = PetscMalloc1(2,&Jx[0]);CHKERRQ(ierr);
  //ierr = PetscMalloc1(2,&Jx[1]);CHKERRQ(ierr);
  //jacobian(1,2,2,x,&Jx);			// TODO: use PetscMalloc1
  jacobian(1,2,2,x,Jx);
  J[0][0] = Jx[0][0];
  J[0][1] = Jx[0][1];
  J[1][0] = Jx[1][0];
  J[1][1] = Jx[1][1];
  free(Jx[0]);
  
  //J[0][0] = 0;
  //J[0][1] = 1.;
  //J[1][0] = -2.*mu*x[1]*x[0]-1.;
  //J[1][1] = mu*(1.0-x[0]*x[0]);

  //printf("J_{exact} =\n    [%.4f, %.4f]\n    [%.4f, %.4f]\n",J[0][0],J[0][1],J[1][0],J[1][1]);
  //printf("J_{adolc} =\n    [%.4f, %.4f]\n    [%.4f, %.4f]\n\n",Jx[0][0],Jx[0][1],Jx[1][0],Jx[1][1]);
  //printf("J_{adolc} =\n    [%.4f, %.4f]\n    [%.4f, %.4f]\n\n",Jx[0],Jx[1],Jx[2],Jx[3]);

  //J[0][0] = Jx[0];
  //J[0][1] = Jx[1];
  //J[1][0] = Jx[2];
  //J[1][1] = Jx[3];

  ierr = MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  //ierr = PetscFree(Jx);CHKERRQ(ierr);		// TODO: use PetscMalloc1

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;	/* ##### Need context ##### */
  PetscReal         mu   = user->mu;	/* ##### Need param ##### */
  PetscInt          row[] = {0,1},col[]={0};
  PetscScalar       J[2][1];
  // PetscScalar       *Jp;			// TODO: use PetscMalloc1
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  PetscScalar** Jp = (PetscScalar**) malloc(2*sizeof(PetscScalar*));
  Jp[0] = (PetscScalar*)malloc(1*sizeof(PetscScalar));
  Jp[1] = (PetscScalar*)malloc(1*sizeof(PetscScalar));
  
  //ierr = PetscMalloc1(2,&Jp);CHKERRQ(ierr);	// TODO: use PetscMalloc1
  //jacobian(2,2,1,&mu,&Jp);			// TODO: use PetscMalloc1
  jacobian(2,2,1,&mu,Jp);			// ##### Evaluate Jacobian using ADOL-C #####

  J[0][0] = 0;
  J[1][0] = (1.-x[0]*x[0])*x[1];

  // ##### TESTING: evaluate exact Jacobian #####
  printf("J_{exact} = [%.4f, %.4f]\n",J[0][0],J[1][0]);
  printf("J_{adolc} = [%.4f, %.4f]\n",Jp[0][0],Jp[1][0]);
  //printf("J_{adolc} = [%.4f, %.4f]\n",Jp[0],Jp[1]);
  printf("mu = %.4f\n\n",mu);

  //J[0][0] = Jp[0];
  //J[1][0] = Jp[1];

  free(Jp);	// ##### Free memory associated with JacobianP #####

  // ierr = PetscFree(Jp);CHKERRQ(ierr);	// TODO: use PetscMalloc1
  ierr = MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscReal         tfinal, dt, tprev;
  User              user = (User)ctx;

  PetscFunctionBeginUser;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&tfinal);CHKERRQ(ierr);
  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",(double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),(double)PetscRealPart(x[1]));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"t %.6f (tprev = %.6f) \n",(double)t,(double)tprev);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  Vec            x;             /* solution, residual vectors */
  Mat            A;             /* Jacobian matrix */
  Mat            Jacp;          /* JacobianP matrix */
  PetscInt       steps;
  PetscReal      ftime   =0.5;
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscErrorCode ierr;
  Vec            lambda[2],mu[2];

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.mu          = 1;
  user.next_output = 0.0;


  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(Jacp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&user);CHKERRQ(ierr);
  /*   Set RHS Jacobian for the adjoint integration */
  ierr = TSSetRHSJacobian(ts,A,A,RHSJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(x,&x_ptr);CHKERRQ(ierr);

  x_ptr[0] = 2;   x_ptr[1] = 0.66666654321;
  ierr = VecRestoreArray(x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.001);CHKERRQ(ierr);

  /*
    Have the TS save its trajectory so that TSAdjointSolve() may be used
  */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,steps,(double)ftime);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Start the Adjoint model
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateVecs(A,&lambda[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&lambda[1],NULL);CHKERRQ(ierr);
  /*   Reset initial conditions for the adjoint integration */
  ierr = VecGetArray(lambda[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 1.0;   x_ptr[1] = 0.0;
  ierr = VecRestoreArray(lambda[0],&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(lambda[1],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;   x_ptr[1] = 1.0;
  ierr = VecRestoreArray(lambda[1],&x_ptr);CHKERRQ(ierr);

  ierr = MatCreateVecs(Jacp,&mu[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(Jacp,&mu[1],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(mu[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(mu[0],&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(mu[1],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(mu[1],&x_ptr);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,2,lambda,mu);CHKERRQ(ierr);


  /*   Set RHS JacobianP */
  ierr = TSSetRHSJacobianP(ts,Jacp,RHSJacobianP,&user);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = VecView(lambda[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(lambda[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(mu[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(mu[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Jacp);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&mu[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&mu[1]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -monitor 0 -viewer_binary_skip_info -ts_trajectory_dirname ex16adjdir

    test:
      suffix: 2
      args: -monitor 0 -ts_trajectory_type memory

TEST*/

