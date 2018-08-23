#define c11 1.0
#define c12 0
#define c21 2.0
#define c22 1.0
static char help[] = "Performs adjoint sensitivity analysis for the van der Pol equation.\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation DAE equivalent
   Concepts: TS^adjoint sensitivity analysis
   Processors: 1
*/
/* ------------------------------------------------------------------------

   This program solves the van der Pol DAE ODE equivalent
      [ u_1' ] = [      u_2              ]  (2)
      [ u_2' ]   [ mu[(1-u_1^2)u_2-u_1]  ]
   on the domain 0 <= x <= 1, with the boundary conditions
       u_1(0) = 2, u_2(0) = -6.666665432100101e-01,
   and
       mu = 10^6,
   and computes the sensitivities of the final solution w.r.t. initial conditions and parameter \mu with the implicit theta method and its discrete adjoint.

   Notes:
   This code demonstrates the TSAdjoint interface to a DAE system.

   The user provides the implicit right-hand-side function
   [ G(u',u,t) ] = [u' - f(u,t)] = [ u_1'] - [        u_2           ]
                                   [ u_2']   [ mu[(1-u_1^2)u_2-u_1] ]

   and the Jacobian of G (from the PETSc user manual)

              dG   dG
   J(G) = a * -- + --
              du'  du

   and the JacobianP of the explicit right-hand side of (2) f(u,t) ( which is equivalent to -G(0,u,t) ).
   df   [       0         ]
   -- = [                 ]
   dp   [ (1 - u_1^2) u_2 ].

   See ex20.c for more details on the Jacobian.

   Many DAEs can be represented in a general form M u_t = f(u,t).
   Thus both sides of (1) are multiplied by an artificial matrix
   M = [ c11 c12 ]
       [ c21 c22 ]
   to turn (1) into the general form. This operation does not change the solution and it is intended for illustration only.

  ------------------------------------------------------------------------- */
#include <petscts.h>
#include <petsctao.h>
#include <adolc/adolc.h>	// Include ADOL-C
#include "subjacobian.c"	// User-defined function to calculate a submatrix of Jacobian

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;

  /* Sensitivity analysis support */
  PetscInt  steps;
  PetscReal ftime;
  Mat       A;                       /* Jacobian matrix */
  Mat       Jacp;                    /* JacobianP matrix */
  Vec       x,lambda[2],mup[2];  /* adjoint variables */
};

/*
*  User-defined routines
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscScalar mu   = user->mu;
  const PetscScalar *x,*xdot;
  PetscScalar       *f;

  adouble           f_a[2];				// adouble for dependent variables
  adouble           x_a[2],mu_a;			// adouble for independent variables

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  trace_on(1);						// Start of active section
  x_a[0] <<= x[0]; x_a[1] <<= x[1]; mu_a <<= mu;	// Declare independence
  f_a[0] = xdot[0] - x_a[1];
  f_a[1] = c21*(xdot[0]-x_a[1]) + xdot[1] - mu_a*((1.0-x_a[0]*x_a[0])*x_a[1] - x_a[0]);
  f_a[0] >>= f[0]; f_a[1] >>= f[1];			// Declare dependence
  trace_off();						// End of active section

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SubJacobian(TS ts,PetscReal t,Vec X,Mat A,Mat B,PetscInt m,PetscInt row[],PetscInt n,PetscInt col[],PetscInt s,PetscInt indep_cols[],PetscBool rhs,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscScalar mu   = user->mu;
  PetscInt          i;
  PetscScalar       **J;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  const PetscScalar indep_vars[] = {x[0],x[1],mu};  	// Concatenate independent vars
  const PetscScalar *ptr_to_indep = indep_vars;         // TODO: how to do this more generally?

  J = myalloc2(m,s);                                    // Contiguous ADOL-C matrix memory allocation
  subjacobian(1,m,n,s,indep_cols,ptr_to_indep,J);       // Calculate Jacobian using ADOL-C
  for(i=0; i<s; i++)
    indep_cols[i] = i;					// Shift column index subset
  if (rhs == PETSC_TRUE){
    ierr = MatSetValues(A,m,row,s,indep_cols,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  } else {
    ierr = MatSetValues(B,m,row,s,indep_cols,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  }
  myfree2(J);                                           // Free allocated memory

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          row[]        = {0,1},m = 2;		// Dependent variables
  PetscInt          col[]        = {0,1,2},n = 3;	// Independent variables
  PetscInt          indep_cols[] = {0,1},s = 2;		// Relevant independent variables
  PetscScalar       C[2][2];
  Mat               M;					// Mass matrix

  PetscFunctionBeginUser;
  ierr = SubJacobian(ts,t,X,A,B,m,row,n,col,s,indep_cols,PETSC_FALSE,ctx);CHKERRQ(ierr);
/*
  For this Jacobian we may use the rule that
     G = M*xdot - f(x)    ==>     dG/dx = a*M - df/dx,
  where a = d(xdot)/dx is a constant.
*/
  ierr = MatCreate(PETSC_COMM_WORLD,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(M);CHKERRQ(ierr);
  ierr = MatSetUp(M);CHKERRQ(ierr);
  C[0][0] = c11;C[0][1] = c12;C[1][0] = c21;C[1][1] = c22;
  ierr = MatSetValues(M,m,row,s,indep_cols,&C[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAXPY(A,a,M,DIFFERENT_NONZERO_PATTERN);
  ierr = MatDestroy(&M);

  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          row[] = {0,1},m = 2;
  PetscInt          col[] = {0,1,2},n = 3;
  PetscInt          indep_cols[]={2},s = 1;

  PetscFunctionBeginUser;
  ierr = SubJacobian(ts,t,X,A,A,m,row,n,col,s,indep_cols,PETSC_TRUE,ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscReal         tfinal, dt;
  User              user = (User)ctx;
  Vec               interpolatedX;

  PetscFunctionBeginUser;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&tfinal);CHKERRQ(ierr);

  while (user->next_output <= t && user->next_output <= tfinal) {
    ierr = VecDuplicate(X,&interpolatedX);CHKERRQ(ierr);
    ierr = TSInterpolate(ts,user->next_output,interpolatedX);CHKERRQ(ierr);
    ierr = VecGetArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",
                       user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),
                       (double)PetscRealPart(x[1]));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = VecDestroy(&interpolatedX);CHKERRQ(ierr);
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar    *x_ptr,*y_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscErrorCode ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output = 0.0;
  user.mu          = 1.0e6;
  user.steps       = 0;
  user.ftime       = 0.5;
  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.A);CHKERRQ(ierr);
  ierr = MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.A);CHKERRQ(ierr);
  ierr = MatSetUp(user.A);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.x,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jacp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user.A,user.A,IJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,user.ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;   x_ptr[1] = -0.66666654321;
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.0001);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,user.x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&user.ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&user.steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreateVecs(user.A,&user.lambda[0],NULL);CHKERRQ(ierr);
  /*   Set initial conditions for the adjoint integration */
  ierr = VecGetArray(user.lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 1.0; y_ptr[1] = 0.0;
  ierr = VecRestoreArray(user.lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.lambda[1],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(user.lambda[1],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 0.0; y_ptr[1] = 1.0;
  ierr = VecRestoreArray(user.lambda[1],&y_ptr);CHKERRQ(ierr);

  ierr = MatCreateVecs(user.Jacp,&user.mup[0],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(user.mup[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(user.mup[0],&x_ptr);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.mup[1],NULL);CHKERRQ(ierr);
  ierr = VecGetArray(user.mup[1],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(user.mup[1],&x_ptr);CHKERRQ(ierr);

  ierr = TSSetCostGradients(ts,2,user.lambda,user.mup);CHKERRQ(ierr);

  /*   Set RHS JacobianP */
  ierr = TSSetRHSJacobianP(ts,user.Jacp,RHSJacobianP,&user);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[y(tf)]/d[y0]  d[y(tf)]/d[z0]\n");CHKERRQ(ierr);
  ierr = VecView(user.lambda[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt initial conditions: d[z(tf)]/d[y0]  d[z(tf)]/d[z0]\n");CHKERRQ(ierr);
  ierr = VecView(user.lambda[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensitivity wrt parameters: d[y(tf)]/d[mu]\n");CHKERRQ(ierr);
  ierr = VecView(user.mup[0],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n sensivitity wrt parameters: d[z(tf)]/d[mu]\n");CHKERRQ(ierr);
  ierr = VecView(user.mup[1],PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Jacp);CHKERRQ(ierr);
  ierr = VecDestroy(&user.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.mup[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.mup[1]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return(ierr);
}

/*TEST

    test:
      requires: revolve
      args: -monitor 0 -ts_type theta -ts_theta_endpoint -ts_theta_theta 0.5 -viewer_binary_skip_info -ts_dt 0.001 -mu 100000 -ts_trajectory_dirname ex20adj1dir

    test:
      suffix: 2
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_solution_only

    test:
      suffix: 3
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_solution_only 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 4
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 5
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 6
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 7
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 8
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_solution_only -ts_trajectory_monitor
      output_file: output/ex20adj_3.out

    test:
      suffix: 9
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_solution_only 0 -ts_trajectory_monitor
      output_file: output/ex20adj_4.out

    test:
      requires: revolve
      suffix: 10
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_revolve_online -ts_trajectory_solution_only
      output_file: output/ex20adj_2.out

    test:
      requires: revolve
      suffix: 11
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 5 -ts_trajectory_revolve_online -ts_trajectory_solution_only 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 12
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_solution_only
      output_file: output/ex20adj_2.out

    test:
      suffix: 13
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_solution_only 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 14
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 15
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 16
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 17
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 18
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 19
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack
      output_file: output/ex20adj_2.out

    test:
      suffix: 20
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_solution_only 0
      output_file: output/ex20adj_2.out

    test:
      suffix: 21
      requires: revolve
      args: -ts_type cn -ts_dt 0.001 -mu 100000 -ts_max_steps 15 -ts_trajectory_type memory -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5 -ts_trajectory_solution_only 0 -ts_trajectory_save_stack 0
      output_file: output/ex20adj_2.out

TEST*/
