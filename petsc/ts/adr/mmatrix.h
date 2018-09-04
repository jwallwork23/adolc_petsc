#ifndef MMATRIX_H
#define MMATRIX_H

#include <vector>

class MMatrix
{
public:
    // constructors
    MMatrix() : nRows(0), nCols(0) {}
    MMatrix(int m,int n,double x=0) : nRows(m), nCols(n), A(m*n,x) {}

    // set all values equal to a double
    MMatrix &operator=(double x)
    {
      for (int i=0; i<nRows*nCols; i++) A[i] = x;
      return *this;
    }

    // access element, indexed by (row,col) [rvalue]
    double operator()(int i, int j) const
    {
      return A[i+j*nRows];
    }

    // access element, indexed by (row,col) [lvalue]
    double &operator()(int i, int j)
    {
      return A[i+j*nRows];
    }

    // size of matrix
    int Rows() const {return nRows;}
    int Cols() const {return nCols;}

private:
    unsigned int nRows, nCols;
    std::vector<double> A;
};

#endif
