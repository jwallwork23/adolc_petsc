#include <petscts.h>
#include <iostream>

using namespace std;

typedef struct {
  double u,v;
} Field;

int main()
{
  int s=0,m=2,i,j;

  Field* a = new Field[m*m];

  for (j=s; j<m*m; j++) {
    cout << &a[j] << ", ";
  }
  cout << endl << endl;

  Field** f = new Field*[m];
  for (j=s; j<m; j++) {
    f[j] = new Field[m];   
    f[j] = a + j*m;		// align with 1-array
    for (i=s; i<m; i++) {
       cout << &f[j][i] << ", ";
    }
    cout << endl;
  }
  cout << endl;
/*
  delete[] a;
  cout << "a deleted" << endl;
*/
  for (j=s; j<m; j++) {
    f[j] -= j*m;
    for (i=s; i<m; i++) {
       cout << &f[j][i] << ", ";
    }
    cout << endl;
    delete[] f[j];
    cout << "f[" << j << "] deleted" << endl;
  }
  delete[] f;
  cout << "f deleted" << endl;

  delete[] a;
  cout << "a deleted" << endl;

  return 0;
}
