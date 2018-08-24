/*
  Calculate a mod b, ensuring a nonnegative result.
*/
int modulo(int a,int b){

  int c = ((a % b) + b) % b;

  return c;
}

/*
  Input: indices i,j,dof from an mxn array of structs, each containing dofs degrees of freedom.
  Output: index k from the corresponding 1-D array.
*/
int coord_map(int i,int j,int dof,int n,int dofs){

  int k = dofs * (n * j + i) + dof;

  return k;
}

/*
  Do coordinate mapping with modular arithmetic
*/
int m_map(int i,int j,int dof,int n,int dofs){

  int k = coord_map(modulo(i,n),modulo(j,n),dof,n,dofs);

  return k;
}
