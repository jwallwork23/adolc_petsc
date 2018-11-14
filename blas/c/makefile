all: clean cleandiff mtmv test_tensor

CFLAGS		= -I${TAPENADE_HOME}/ADFirstAidKit -I${LAPACK_HOME}/CBLAS -I${LAPACK_HOME}/LAPACKE -Wall
LIB		= -L${LAPACK_HOME} -lcblas -lrefblas -ltmglib -llapack -llapacke -Wall
DDIR		= derivatives
TFLAGS  	= -O $(DDIR)
CLEANFILES	= test_AB test_A test_tensor *.o *~

clean:
	rm -rf $(CLEANFILES)

cleandiff:
	rm -rf $(DDIR)/*_d.c $(DDIR)/*_d.msg $(DDIR)/*_b.c $(DDIR)/*_b.msg $(DDIR)/*~

### DIFFERENTIATION OF MATRIX / TENSOR SUBROUTINES USING TAPENADE

# Differentiate (naive) Hadamard product C := A.*B
mpm:
	tapenade -tangent -head "naive_mpm (A B)\(C)" -o mpm $(TFLAGS) mxm.c
	tapenade -tangent -head "naive_mpm (A)\(C)" -o mpm_A $(TFLAGS) -tgtfuncname _A_d mxm.c
	tapenade -tangent -head "naive_mpm (B)\(C)" -o mpm_B $(TFLAGS) -tgtfuncname _B_d mxm.c
	tapenade -reverse -head "naive_mpm (A B)\(C)" -o mpm $(TFLAGS) mxm.c
	tapenade -reverse -head "naive_mpm (A)\(C)" -o mpm_A $(TFLAGS) -adjfuncname _A_b mxm.c
	tapenade -reverse -head "naive_mpm (B)\(C)" -o mpm_B $(TFLAGS) -adjfuncname _B_b mxm.c

# Differentiate (naive) matrix product C := A*B
mxm:
	tapenade -tangent -head "naive_mxm (A B)\(C)" -o mxm.c $(TFLAGS) mxm.c
	tapenade -tangent -head "naive_mxm (A)\(C)" -o mxm_A.c $(TFLAGS) -tgtfuncname _A_d mxm.c
	tapenade -tangent -head "naive_mxm (B)\(C)" -o mxm_B.c $(TFLAGS) -tgtfuncname _B_d mxm.c
	tapenade -reverse -head "naive_mxm (A,B)\(C)" -o mxm.c $(TFLAGS) mxm.c
	tapenade -reverse -head "naive_mxm (A)\(C)" -o mxm_A.c $(TFLAGS) -adjfuncname _A_b mxm.c
	tapenade -reverse -head "naive_mxm (B)\(C)" -o mxm_B.c $(TFLAGS) -adjfuncname _B_b mxm.c

# Differentiate (naive) DGEMM implementation C := alpha*op(A)*op(B) + beta*C w.r.t. matrix arguments
dgemm:
	tapenade -tangent -head "naive_dgemm (A B)\(C)" $(TFLAGS) mxm.c
	tapenade -tangent -head "naive_dgemm (A)\(C)" $(TFLAGS) -tgtfuncname _A_d mxm.c
	tapenade -tangent -head "naive_dgemm (B)\(C)" $(TFLAGS) -tgtfuncname _B_d mxm.c
	tapenade -reverse -head "naive_dgemm (A B)\(C)" $(TFLAGS) mxm.c
	tapenade -reverse -head "naive_dgemm (A)\(C)" $(TFLAGS) -adjfuncname _A_b mxm.c
	tapenade -reverse -head "naive_dgemm (B)\(C)" $(TFLAGS) -adjfuncname _B_b mxm.c

# Differentiate (naive) DGEMM implementation C := alpha*op(A)*op(B) + beta*C w.r.t. scalar arguments
dgemm_scalar:
	tapenade -tangent -head "naive_dgemm (alpha beta)\(C)" $(TFLAGS) -tgtfuncname _s_d mxm.c
	tapenade -reverse -head "naive_dgemm (alpha beta)\(C)" $(TFLAGS) -adjfuncname _s_b mxm.c

# Differentiate 'matrix-tensor-matrix-vector' product vec(V) = (A \otimes B) vec(U) <=> V = B*U*A^T
#  w.r.t. matrix arguments
mtmv:
	tapenade -tangent -head "naive_mtmv (U)\(V)" $(TFLAGS) mxm.c
	tapenade -reverse -head "naive_mtmv (U)\(V)" $(TFLAGS) mxm.c

### TAPENADE UTILS

adBuffer.o: ${TAPENADE_HOME}/ADFirstAidKit/adBuffer.c
	gcc $(CFLAGS) -c -o $@ $^

adStack.o: ${TAPENADE_HOME}/ADFirstAidKit/adStack.c
	gcc $(CFLAGS) -c -o $@ $^

### SIMPLE TESTS

test_AB.o: test_AB.c
	gcc $(CFLAGS) -c -o $@ $^

test_AB: test_AB.o adBuffer.o adStack.o
	gfortran -o $@ $^ $(LIB)
	rm *.o

test_A.o: test_A.c
	gcc $(CFLAGS) -c -o $@ $^

test_A: test_A.o adBuffer.o adStack.o
	gfortran -o $@ $^ $(LIB)
	rm *.o

test_tensor.o: test_tensor.c
	gcc $(CFLAGS) -c -o $@ $^

test_tensor: test_tensor.o adBuffer.o adStack.o
	gfortran -o $@ $^ $(LIB)
	rm *.o
