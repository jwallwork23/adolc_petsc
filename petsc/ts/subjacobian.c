#include <adolc/adolc.h>

/*--------------------------------------------------------------------------*/
/*                                                                 jacobian */
/* subjacobian(tag, m, n, s, var_select[s], x[n], J[m][s])                  */

int subjacobian(short tag,
                int depen,
                int indep,
                int subs,			// number of independent variables to consider
                int *subset,			// subset of independent variables
                const double *argument,
                double **jacobian) {
    int rc,i,j;
    double *result, **I;

    result = myalloc1(depen);
/*
  Instead of postmultiplying in fov_forward with an identity matrix (i.e. computing the full
  Jacobian), we postmultiply by an m x s submatrix thereof, where s<=n. That is, we may compute the
  Jacobian w.r.t. a subset of the independent variables.
*/
    I = myalloc2(indep,subs);
    for(i=0; i<indep; i++){
        for(j=0; j<subs; j++){
            if(i == subset[j])
                I[i][j] = 1.;
        }
    }
    rc = fov_forward(tag,depen,indep,subs,argument,I,result,jacobian);
    myfree2(I);

/* 
  TODO: Also consider the case of calculating Jacobian of a subset of the dependent variables, using
        fov_reverse.
*/

    myfree1(result);

    return rc;
}

