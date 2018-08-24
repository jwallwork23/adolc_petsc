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
int coord_mapping(int i,int j,int dof,int n,int dofs){

  int k = dofs * (n * j + i) + dof;

  return k;
}
