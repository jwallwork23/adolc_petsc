#include <math.h>

/* MMS1: Method of Manufactured Solutions

  u = t + x^2 + y^2;
  v = t + 2*x^2 - 2*x*y;
  p = x + y - 1;

  f_x = -2*t*(x + y) + 2*x*y^2 - 4*x^2*y - 2*x^3 + 4.0/Re - 1.0
  f_y = -2*t*x       + 2*y^3 - 4*x*y^2 - 2*x^2*y + 4.0/Re - 1.0

  so that

    u_t + u \cdot \nabla u - 1/Re \Delta u + \nabla p + f = <1, 1> + <t (2x + 2y) + 2x^3 + 4x^2y - 2xy^2, t 2x + 2x^2y + 4xy^2 - 2y^3> - 1/Re <4, 4> + <1, 1>
                                                    + <-t (2x + 2y) + 2xy^2 - 4x^2y - 2x^3 + 4/Re - 1, -2xt + 2y^3 - 4xy^2 - 2x^2y + 4/Re - 1> = 0
    \nabla \cdot u                                  = 2x - 2x = 0

  where

    <u, v> . <<u_x, v_x>, <u_y, v_y>> = <u u_x + v u_y, u v_x + v v_y>
*/
int mms1_u_2d(int dim, double time, const double x[], int Nf, double *u, void *ctx)
{
  u[0] = time + x[0]*x[0] + x[1]*x[1];
  u[1] = time + 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
  return 0;
}

int mms1_p_2d(int dim, double time, const double x[], int Nf, double *p, void *ctx)
{
  *p = x[0] + x[1] - 1.0;
  return 0;
}

/* MMS 2: Alternative Method of Manufactured Solutions */

static int mms2_u_2d(int dim, double time, const double x[], int Nf, double *u, void *ctx)
{
  u[0] = sin(time + x[0])*sin(time + x[1]);
  u[1] = cos(time + x[0])*cos(time + x[1]);
  return 0;
}

static int mms2_p_2d(int dim, double time, const double x[], int Nf, double *p, void *ctx)
{
  *p = sin(time + x[0] - x[1]);
  return 0;
}
