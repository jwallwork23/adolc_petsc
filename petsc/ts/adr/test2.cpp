#include <petscts.h>
#include <adolc/adolc.h>
#include <iostream>

using namespace std;

// TODO: Continue along this avenue. See determinant example

typedef struct {
  adouble u,v;
} aField;

aField **A;

int main()
{
  int n,i,j;
  double a = 2.,b=1.;
  adouble tmp;

  cout << "Matrix dimension ?= ";
  cin >> n;
  cout << endl;

  cout << "Before: a = " << a << ", b = " << b << endl;

  A = new aField*[n];
  trace_on(1);
  for (i=0; i<n; i++) {
    A[i] = new aField[n];
    for (j=0; j<n; j++) {
      A[i][j].u <<= a;A[i][j].v <<= b;
      tmp = A[i][j].u;A[i][j].u = A[i][j].v;A[i][j].v = tmp;
      A[i][j].u >>= a;A[i][j].v >>= b;
    }
  }
  trace_off();

  cout << "After: a = " << a << ", b = " << b << endl;

  delete[] A;

  return 0;
}
