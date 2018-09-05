#include <petscts.h>
#include <iostream>

using namespace std;

typedef struct {
  PetscScalar u,v;
} Field;

int main()
{
  // 1-array
  Field* e = new Field[4];

  // 2-array
  Field** d = new Field*[2];
  d[0] = new Field[2];
  d[1] = new Field[2];

  cout << "Initial addresses:" << endl;
  cout << "&d[0][0] = " << &d[0][0] << endl;
  cout << "&d[0][1] = " << &d[0][1] << endl;
  cout << "&d[1][0] = " << &d[1][0] << endl;
  cout << "&d[1][1] = " << &d[1][1] << endl;
  cout << "&e[0] = " << &e[0] << endl;
  cout << "&e[1] = " << &e[1] << endl;
  cout << "&e[2] = " << &e[2] << endl;
  cout << "&e[3] = " << &e[3] << endl << endl;

  // set value in 2-array to value in 1-array
  delete[] d[0];
  d[0] = e;
  delete[] d[1];
  d[1] = e + 2;

  cout << "Addresses after assignment:" << endl;
  cout << "&d[0][0] = " << &d[0][0] << endl;
  cout << "&d[0][1] = " << &d[0][1] << endl;
  cout << "&d[1][0] = " << &d[1][0] << endl;
  cout << "&d[1][1] = " << &d[1][1] << endl;
  cout << "&e[0] = " << &e[0] << endl;
  cout << "&e[1] = " << &e[1] << endl;
  cout << "&e[2] = " << &e[2] << endl;
  cout << "&e[3] = " << &e[3] << endl << endl;

  delete[] d;
  delete[] e;

  return 0;
}
