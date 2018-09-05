#include <petscts.h>
#include <iostream>
#include <adolc/adolc.h>

using namespace std;

typedef struct {
  PetscScalar u,v;
} Field;

typedef struct {
  adouble     u,v;
} aField;

void FieldPrint(Field **f,PetscInt s,PetscInt m)
{
    PetscInt i,j;

    cout << endl << "u component:" << endl;
    for (j = s; j < m; j++) {
        for (i = s; i < m; i++) {
            cout << f[j][i].u << ", ";
        }
        cout << endl;
    }
    cout << "v component:" << endl;
    for (j = s; j < m; j++) {
        for (i = s; i < m; i++) {
            cout << f[j][i].v << ", ";
        }
        cout << endl;
    }
    cout << endl;
}

int main(int argc,char **argv)
{
    PetscErrorCode ierr;
    PetscInt       ss=0,i,j,s=0,m=5,dof=2,gs=0,gm=5;
    PetscBool      ghost;

    ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
    ierr = PetscOptionsGetBool(NULL,NULL,"-ghost",&ghost,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

    //Create an array of pointers that points to more arrays of Fields
    Field** matrix1 = new Field*[m];
    for (j = s; j < m; j++) {
        matrix1[j] = new Field[m];
    }

    for (j = s; j < m; j++) {
        for (i = s; i < m; i++) {
            matrix1[j][i].u = (j*m + i) % 2;
            matrix1[j][i].v = (i*m + j) % 2;
        }
    }

    //Do the same for outputs
    Field** matrix2 = new Field*[m];
    for (j = s; j < m; j++) {
        matrix2[j] = new Field[m];
    }

    if (ghost) {
      gs = -1;gm = 7;
    } else {
      ss += 1;
    }

    //Print out input matrices to verify they have been created
    cout << endl << "BEFORE" << endl;
    FieldPrint(matrix1,s,m);

    //Contiguous 1-arrays for active matrices
    aField* alist1 = new aField[gm*gm];
    aField* alist2 = new aField[gm*gm];

    //Create an array of pointers that points to more arrays of aFields
    aField** amatrix1 = new aField*[gm];
    for (j = 0; j < gm; j++) {
        amatrix1[j] = new aField[gm];
        delete[] amatrix1[j];
        amatrix1[j] = alist1 + dof*j*gm - dof*gs;
    }
    *amatrix1 -= gs;

    //Do the same for outputs
    aField** amatrix2 = new aField*[gm];
    for (j = 0; j < gm; j++) {
        amatrix2[j] = new aField[gm];
        delete[] amatrix2[j];
        amatrix2[j] = alist2 + dof*j*gm - dof*gs;
    }
    *amatrix2 -= gs;

    trace_on(1);	/* ------------------- ACTIVE SECTION ----------------------- */

    //Mark independence
    for (j = s; j < m; j++) {
        for (i = s; i < m; i++) {
            amatrix1[j][i].u <<= matrix1[j][i].u;
            amatrix1[j][i].v <<= matrix1[j][i].v;
        }
    }

    //Work on active variables
    for (j=s; j<m; j++) {
      for (i=ss; i<m; i++) {
        amatrix2[j][i].u = amatrix1[j][i-1].u;
      }
    }
    for (j=ss; j<m; j++) {
      for (i=s; i<m; i++) {
        amatrix2[j][i].v = amatrix1[j-1][i].v;
      }
    }

    //Mark dependence
    for (j = s; j < m; j++) {
        for (i = s; i < m; i++) {
            amatrix2[j][i].u >>= matrix2[j][i].u;
            amatrix2[j][i].v >>= matrix2[j][i].v;
        }
    }

    trace_off();	/* ----------------------------------------------------------- */

    //Print out the matrices to verify everything has gone through
    cout << endl;
    FieldPrint(matrix2,s,m);

    //Free memory
    for(j = 0; j < m; j++) {
        delete[] matrix2[j];   
    }
    for(j = 0; j < m; j++) {
        delete[] matrix1[j];   
    }

    delete[] amatrix2;
    delete[] matrix2;
    delete[] amatrix1;
    delete[] matrix1;
    delete[] alist2;
    delete[] alist1;

    ierr = PetscFinalize();CHKERRQ(ierr);

    return ierr;
}
