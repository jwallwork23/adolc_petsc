#include <math.h>

/* Reynolds number */
#define REYN 400.0


static void f0_mms1_u(int dim, int Nf, int NfAux,
                      const int uOff[], const int uOff_x[], const double u[], const double u_t[], const double u_x[],
                      const int aOff[], const int aOff_x[], const double a[], const double a_t[], const double a_x[],
                      double t, const double x[], int numConstants, const double constants[], double f0[dim])
{
  const double Re    = REYN;
  const int    Ncomp = dim;
  int          c, d;

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

static void f0_mms2_u(int dim, int Nf, int NfAux,
                      const int uOff[], const int uOff_x[], const double u[dim], const double u_t[dim], const double u_x[],
                      const int aOff[], const int aOff_x[], const double a[], const double a_t[], const double a_x[],
                      double t, const double x[], int numConstants, const double constants[], double f0[dim])
{
  const double Re    = REYN;
  const int  Ncomp = dim;
  int        c, d;

  for (c = 0; c < Ncomp; ++c) {
    for (d = 0; d < dim; ++d) {
      f0[c] += u[d] * u_x[c*dim+d];
    }
  }
  f0[0] += u_t[0];
  f0[1] += u_t[1];

  f0[0] -= ( Re*((1.0L/2.0L)*sin(2*t + 2*x[0]) + sin(2*t + x[0] + x[1]) + cos(t + x[0] - x[1])) + 2.0*sin(t + x[0])*sin(t + x[1]))/Re;
  f0[1] -= (-Re*((1.0L/2.0L)*sin(2*t + 2*x[1]) + sin(2*t + x[0] + x[1]) + cos(t + x[0] - x[1])) + 2.0*cos(t + x[0])*cos(t + x[1]))/Re;
}

static void f1_u(int dim, int Nf, int NfAux,
                 const int uOff[], const int uOff_x[], const double u[], const double u_t[], const double u_x[],
                 const int aOff[], const int aOff_x[], const double a[], const double a_t[], const double a_x[],
                 double t, const double x[], int numConstants, const double constants[], double f1[dim*dim])
{
  const double Re    = REYN;
  const int  Ncomp = dim;
  int        comp, d;

  for (comp = 0; comp < Ncomp; ++comp) {
    for (d = 0; d < dim; ++d) {
      f1[comp*dim+d] = 1.0/Re * u_x[comp*dim+d];
    }
    f1[comp*dim+comp] -= u[Ncomp];
  }
}

static void f0_p(int dim, int Nf, int NfAux,
                 const int uOff[], const int uOff_x[], const double u[], const double u_t[], const double u_x[],
                 const int aOff[], const int aOff_x[], const double a[], const double a_t[], const double a_x[],
                 double t, const double x[], int numConstants, const double constants[], double f0[1])
{
  int d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

static void f1_p(int dim, int Nf, int NfAux,
                 const int uOff[], const int uOff_x[], const double u[], const double u_t[], const double u_x[],
                 const int aOff[], const int aOff_x[], const double a[], const double a_t[], const double a_x[],
                 double t, const double x[], int numConstants, const double constants[], double f1[dim])
{
  int d;
  for (d = 0; d < dim; ++d) f1[d] = 0.0;
}

