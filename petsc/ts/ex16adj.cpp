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
#include <adolc/adolc.h>	// Include ADOL-C

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
 
  adouble           f_a[2];   				// adouble for dependent variables
  adouble           x_a[2];   				// adouble for independent variables
  adouble           mu_a;				// adouble for mu parameter

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);      	// Get values for passive indep variables
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);          	// Get array for passive dependent variables

  trace_on(1);						// Start of active section
  x_a[0] <<= x[0]; x_a[1] <<= x[1]; mu_a <<= mu;	// Declare independence
  f_a[0] = x_a[1];
  f_a[1] = mu_a*(1.-x_a[0]*x_a[0])*x_a[1]-x_a[0];
  f_a[0] >>= f[0]; f_a[1] >>= f[1];			// Mark dependence
  trace_off(1);						// End of active section

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);  	// Restore passive indep. variable array
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);      	// Give values to passive dep. varible array
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSSubJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,PetscInt indep_cols[],void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscReal   mu   = user->mu;
  PetscInt          row[] = {0,1},col[] = {0,1,2},i,j;
  PetscInt          m = sizeof(row)/sizeof(row[0]);	// Number of dependent variables
  PetscInt          n = sizeof(col)/sizeof(col[0]);	// Number of independent variables
  PetscInt          s = sizeof(indep_cols)/sizeof(indep_cols[0]);
  PetscScalar       J[m][s];
  PetscScalar       **Jx;
  PetscScalar       *row0,*row1;			// TODO: how to do this more generally?
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  const PetscScalar indep_vars[3] = {x[0],x[1],mu};	// Concatenate independent vars
  const PetscScalar *ptr_to_indep = indep_vars;		// Give a pointer

  ierr = PetscMalloc1(m,&Jx);CHKERRQ(ierr);		// Allocate memory for Jacobian
  ierr = PetscMalloc1(n,&row0);CHKERRQ(ierr);		// TODO: how to do this more generally?
  ierr = PetscMalloc1(n,&row1);CHKERRQ(ierr);		// TODO: ------------"-----------------
  Jx[0] = row0; Jx[1] = row1;				// TODO: ------------"-----------------
  jacobian(1,m,n,ptr_to_indep,Jx);			// Calculate Jacobian using ADOL-C
  for(i=0; i<m; i++){
    for(j=0; j<s; j++){
      J[i][j] = Jx[i][indep_cols[j]];
    }
  }
  for(i=0; i<s; i++)					// Shift column index subset
    indep_cols[i] = i;
  ierr = MatSetValues(A,m,row,s,indep_cols,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = PetscFree(Jx);CHKERRQ(ierr);
  ierr = PetscFree(row0);CHKERRQ(ierr);			// TODO: how to do this more generally?
  ierr = PetscFree(row1);CHKERRQ(ierr);			// TODO: ------------"-----------------
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          indep_cols[] = {0,1};		// Choose relevant independent variables

  PetscFunctionBeginUser;
  ierr = RHSSubJacobian(ts,t,X,A,B,indep_cols,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
/*
  User              user = (User)ctx;
  const PetscReal   mu   = user->mu;
  PetscInt          row[] = {0,1},col[] = {0,1,2},i,j;
  PetscInt          m = sizeof(row)/sizeof(row[0]);	// Number of dependent variables
  PetscInt          n = sizeof(col)/sizeof(col[0]);	// Number of independent variables
  PetscInt          s = sizeof(indep_cols)/sizeof(indep_cols[0]);
  PetscScalar       J[m][s];
  PetscScalar       **Jx;
  PetscScalar       *row0,*row1;			// TODO: how to do this more generally?
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  const PetscScalar indep_vars[3] = {x[0],x[1],mu};	// Concatenate independent vars
  const PetscScalar *ptr_to_indep = indep_vars;		// Give a pointer

  ierr = PetscMalloc1(m,&Jx);CHKERRQ(ierr);		// Allocate memory for Jacobian
  ierr = PetscMalloc1(n,&row0);CHKERRQ(ierr);		// TODO: how to do this more generally?
  ierr = PetscMalloc1(n,&row1);CHKERRQ(ierr);		// TODO: ------------"-----------------
  Jx[0] = row0; Jx[1] = row1;				// TODO: ------------"-----------------
  jacobian(1,m,n,ptr_to_indep,Jx);			// Calculate Jacobian using ADOL-C
  for(i=0; i<m; i++){
    for(j=0; j<s; j++){
      J[i][j] = Jx[i][indep_cols[j]];
    }
  }
  for(i=0; i<s; i++)					// Shift column index subset
    indep_cols[i] = i;
  ierr = MatSetValues(A,m,row,s,indep_cols,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = PetscFree(Jx);CHKERRQ(ierr);
  ierr = PetscFree(row0);CHKERRQ(ierr);			// TODO: how to do this more generally?
  ierr = PetscFree(row1);CHKERRQ(ierr);			// TODO: ------------"-----------------
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
*/
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx)
{

  PetscErrorCode    ierr;
  PetscInt          indep_cols[] = {2};
/*
  PetscFunctionBeginUser;				// TODO: why does this version not work?
  ierr = RHSSubJacobian(ts,t,X,A,A,indep_cols,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
*/

  User              user = (User)ctx;
  PetscReal         mu   = user->mu;
  PetscInt          row[] = {0,1},col[] = {0,1,2},i,j;
  PetscInt          m = sizeof(row)/sizeof(row[0]);	// Number of dependent variables
  PetscInt          n = sizeof(col)/sizeof(col[0]);	// Number of independent variables
  PetscInt          s = sizeof(indep_cols)/sizeof(indep_cols[0]);
  PetscScalar       J[m][s];
  PetscScalar       **Jp;
  PetscScalar       *row0,*row1;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  const PetscScalar indep_vars[3] = {x[0],x[1],mu};	// Concatenate independent vars
  const PetscScalar *ptr_to_indep = indep_vars;		// Give a pointer

  ierr = PetscMalloc1(m,&Jp);CHKERRQ(ierr);		// Allocate memory for Jacobian
  ierr = PetscMalloc1(n,&row0);CHKERRQ(ierr);
  ierr = PetscMalloc1(n,&row1);CHKERRQ(ierr);
  Jp[0] = row0; Jp[1] = row1;
  jacobian(1,m,n,ptr_to_indep,Jp);			// Calculate Jacobian using ADOL-C
  for(i=0; i<m; i++){
    for(j=0; j<s; j++){
      J[i][j] = Jp[i][indep_cols[j]];
    }
  }
  for(i=0; i<s; i++)					// Shift column index subset
    indep_cols[i] = i;
  ierr = MatSetValues(A,m,row,s,indep_cols,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = PetscFree(Jp);CHKERRQ(ierr);			// Free allocated memory
  ierr = PetscFree(row0);CHKERRQ(ierr);
  ierr = PetscFree(row1);CHKERRQ(ierr);
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

