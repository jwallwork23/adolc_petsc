/* Reynolds number */
#define REYN 400.0

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
PetscErrorCode mms1_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = time + x[0]*x[0] + x[1]*x[1];
  u[1] = time + 2.0*x[0]*x[0] - 2.0*x[0]*x[1];
  return 0;
}

PetscErrorCode mms1_p_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  *p = x[0] + x[1] - 1.0;
  return 0;
}

/* MMS 2: Alternative Method of Manufactured Solutions */

static PetscErrorCode mms2_u_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  u[0] = PetscSinReal(time + x[0])*PetscSinReal(time + x[1]);
  u[1] = PetscCosReal(time + x[0])*PetscCosReal(time + x[1]);
  return 0;
}

static void f0_mms1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal Re    = REYN;
  const PetscInt  Ncomp = dim;
  PetscInt        c, d;

  for (c = 0; c < Ncomp; ++c) {
    for (d = 0; d < dim; ++d) {
      f0[c] += u[d] * u_x[c*dim+d];
    }
  }
  f0[0] += u_t[0];
  f0[1] += u_t[1];

  f0[0] += -2.0*t*(x[0] + x[1]) + 2.0*x[0]*x[1]*x[1] - 4.0*x[0]*x[0]*x[1] - 2.0*x[0]*x[0]*x[0] + 4.0/Re - 1.0;
  f0[1] += -2.0*t*x[0]          + 2.0*x[1]*x[1]*x[1] - 4.0*x[0]*x[1]*x[1] - 2.0*x[0]*x[0]*x[1] + 4.0/Re - 1.0;
}

static void f0_mms2_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal Re    = REYN;
  const PetscInt  Ncomp = dim;
  PetscInt        c, d;

  for (c = 0; c < Ncomp; ++c) {
    for (d = 0; d < dim; ++d) {
      f0[c] += u[d] * u_x[c*dim+d];
    }
  }
  f0[0] += u_t[0];
  f0[1] += u_t[1];

  f0[0] -= ( Re*((1.0L/2.0L)*PetscSinReal(2*t + 2*x[0]) + PetscSinReal(2*t + x[0] + x[1]) + PetscCosReal(t + x[0] - x[1])) + 2.0*PetscSinReal(t + x[0])*PetscSinReal(t + x[1]))/Re;
  f0[1] -= (-Re*((1.0L/2.0L)*PetscSinReal(2*t + 2*x[1]) + PetscSinReal(2*t + x[0] + x[1]) + PetscCosReal(t + x[0] - x[1])) + 2.0*PetscCosReal(t + x[0])*PetscCosReal(t + x[1]))/Re;
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal Re    = REYN;
  const PetscInt  Ncomp = dim;
  PetscInt        comp, d;

  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = 1.0/Re * u_x[comp*dim+d];
    }
    f1[comp*dim+comp] -= u[Ncomp];
  }
}

static void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

static void f1_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = 0.0;
}

