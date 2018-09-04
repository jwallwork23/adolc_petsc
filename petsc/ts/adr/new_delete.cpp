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
    int            xss=0,yss=0,i,j;
    PetscBool      ghost;

    ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
    ierr = PetscOptionsGetBool(NULL,NULL,"-ghost",&ghost,NULL);CHKERRQ(ierr);


    //Create an array of pointers that points to more arrays of Fields
    Field** matrix1 = new Field*[5];
    for (j = 0; j < 5; j++) {
        matrix1[j] = new Field[5];
    }

    for (j = 0; j < 5; j++) {
        for (i = 0; i < 5; i++) {
            matrix1[j][i].u = (j*5 + i) % 2;
            matrix1[j][i].v = (i*5 + j) % 2;
        }
    }

    //Print out the matrices to verify we have created the matrix
    cout << endl << "BEFORE" << endl << endl << "u component:" << endl;
    for (j = 0; j < 5; j++) {
        for (i = 0; i < 5; i++) {
            cout << matrix1[j][i].u << ", ";
        }
        cout << endl;
    }
    cout << "v component:" << endl;
    for (j = 0; j < 5; j++) {
        for (i = 0; i < 5; i++) {
            cout << matrix1[j][i].v << ", ";
        }
        cout << endl;
    }
    cout << endl;

    //Create an array of pointers that points to more arrays of aFields
    aField** amatrix1 = new aField*[5];
    for (j = 0; j < 5; j++) {
        amatrix1[j] = new aField[5];
    }

    //Do the same for outputs
    Field** matrix2 = new Field*[5];
    for (j = 0; j < 5; j++) {
        matrix2[j] = new Field[5];
    }

    //Do the same for outputs
    aField** amatrix2 = new aField*[5];
    for (j = 0; j < 5; j++) {
        amatrix2[j] = new aField[5];
    }

    trace_on(1);

    for (j = 0; j < 5; j++) {
        for (i = 0; i < 5; i++) {
            amatrix1[j][i].u <<= matrix1[j][i].u;
            amatrix1[j][i].v <<= matrix1[j][i].v;
        }
    }

    if (!ghost) {
      xss += 1;yss += 1;
    }
    for (j=0; j<5; j++) {
      for (i=xss; i<5; i++) {
        amatrix2[j][i].u = amatrix1[j][i-1].u;
      }
    }
    for (j=yss; j<5; j++) {
      for (i=0; i<5; i++) {
        amatrix2[j][i].v = amatrix1[j-1][i].v;
      }
    }

    for (j = 0; j < 5; j++) {
        for (i = 0; i < 5; i++) {
            amatrix2[j][i].u >>= matrix2[j][i].u;
            amatrix2[j][i].v >>= matrix2[j][i].v;
        }
    }

    trace_off();

    //Print out the matrices to verify everything has gone through
    cout << endl << "AFTER" << endl << endl << "u component:" << endl;
    for (j = 0; j < 5; j++) {
        for (i = 0; i < 5; i++) {
            cout << matrix2[j][i].u << ", ";
        }
        cout << endl;
    }
    cout << "v component:" << endl;
    for (j = 0; j < 5; j++) {
        for (i = 0; i < 5; i++) {
            cout << matrix2[j][i].v << ", ";
        }
        cout << endl;
    }
    cout << endl;

    //Free each sub-array
    for(j = 0; j < 5; j++) {
        delete[] amatrix2[j];   
    }
    //Free the array of pointers
    delete[] amatrix2;

    //Free each sub-array
    for(j = 0; j < 5; j++) {
        delete[] matrix2[j];   
    }
    //Free the array of pointers
    delete[] matrix2;

    //Free each sub-array
    for(j = 0; j < 5; j++) {
        delete[] amatrix1[j];   
    }
    //Free the array of pointers
    delete[] amatrix1;

    //Free each sub-array
    for(j = 0; j < 5; j++) {
        delete[] matrix1[j];   
    }
    //Free the array of pointers
    delete[] matrix1;

    ierr = PetscFinalize();CHKERRQ(ierr);

    return ierr;
}
