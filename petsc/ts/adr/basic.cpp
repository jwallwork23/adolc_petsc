#include <petscts.h>
#include <iostream>

using namespace std;

int main()
{

  double** d = new double*[1];
  d[0] = new double[1];
  double* e = new double[1];

  cout << "Initial addresses:" << endl;
  cout << "&d = " << &d[0][0] << endl;
  cout << "&e = " << &e[0] << endl << endl;

  d[0] = e;

  cout << "Addresses after assignment:" << endl;
  cout << "&d = " << &d[0][0] << endl;
  cout << "&e = " << &e[0] << endl;

  delete[] e;
  //delete[] d[0];
  delete[] d;

  return 0;
}
