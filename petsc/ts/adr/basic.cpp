#include <petscts.h>
#include <iostream>

using namespace std;

typedef struct {
  PetscScalar u,v;
} Field;

int main()
{
  // 1-array
  Field* e = new Field[1];

  // 2-array
  Field** d = new Field*[1];
  d[0] = new Field[1];

  cout << "Initial addresses:" << endl;
  cout << "&d = " << &d[0][0] << endl;
  cout << "&e = " << &e[0] << endl << endl;

  // set value in 2-array to value in 1-array
  delete[] d[0];
  d[0] = e;

  cout << "Addresses after assignment:" << endl;
  cout << "&d = " << &d[0][0] << endl;
  cout << "&e = " << &e[0] << endl;

  delete[] d;
  delete[] e;

  return 0;
}
