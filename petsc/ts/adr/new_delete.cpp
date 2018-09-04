#include <petscts.h>
#include <iostream>
#include <adolc/adolc.h>

using namespace std;

typedef struct {
  double u,v;
} Field;

typedef struct {
  adouble u,v;
} aField;

int main(int argc,char **argv)
{
    PetscErrorCode ierr;

    ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;

    //Create an array of pointers that points to more arrays of Fields
    Field** matrix = new Field*[5];
    for (int j = 0; j < 5; j++) {
        matrix[j] = new Field[5];
    }

    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 5; i++) {
            matrix[j][i].u = j*5 + i + 1;
            matrix[j][i].v = i*5 + j + 1;
        }
    }

    //Print out the matrices to verify we have created the matrix
    cout << endl << "BEFORE" << endl << endl << "u component:" << endl;
    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 5; i++) {
            cout << matrix[j][i].u << ", ";
        }
        cout << endl;
    }
    cout << "v component:" << endl;
    for (int j = 0; j < 5; j++) {
        for (int i = 0; i < 5; i++) {
            cout << matrix[j][i].v << ", ";
        }
        cout << endl;
    }
    cout << endl;

    //Create an array of pointers that points to more arrays of aFields
    aField** amatrix = new aField*[5];
    for (int j = 0; j < 5; j++) {
        amatrix[j] = new aField[5];
    }





    //Free each sub-array
    for(int j = 0; j < 5; j++) {
        delete[] matrix[j];   
    }
    //Free the array of pointers
    delete[] matrix;

    //Free each sub-array
    for(int j = 0; j < 5; j++) {
        delete[] amatrix[j];   
    }
    //Free the array of pointers
    delete[] amatrix;

    ierr = PetscFinalize();CHKERRQ(ierr);

    return ierr;
}
