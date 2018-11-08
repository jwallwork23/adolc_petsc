all: clean cleandiff dgemm test_A

CFLAGS		= -Wall -I${TAPENADE_HOME}/ADFirstAidKit -I${LAPACK_HOME}/CBLAS
LIB		= -L${LAPACK_HOME} -lcblas -lrefblas -ltmglib -llapack
TFLAGS  	= -O derivatives
CLEANFILES	= test_AB test_A *.o *~

mpm:
	tapenade -tangent -head "naive_mpm (A B)\(C)" -o mpm $(TFLAGS) mxm.c
	tapenade -tangent -head "naive_mpm (A)\(C)" -o mpm_A $(TFLAGS) -tgtfuncname _A_d mxm.c
	tapenade -tangent -head "naive_mpm (B)\(C)" -o mpm_B $(TFLAGS) -tgtfuncname _B_d mxm.c
	tapenade -reverse -head "naive_mpm (A B)\(C)" -o mpm $(TFLAGS) mxm.c
	tapenade -reverse -head "naive_mpm (A)\(C)" -o mpm_A $(TFLAGS) -adjfuncname _A_b mxm.c
	tapenade -reverse -head "naive_mpm (B)\(C)" -o mpm_B $(TFLAGS) -adjfuncname _B_b mxm.c

mxm:
	tapenade -tangent -head "naive_mxm (A B)\(C)" -o mxm.c $(TFLAGS) mxm.c
	tapenade -tangent -head "naive_mxm (A)\(C)" -o mxm_A.c $(TFLAGS) -tgtfuncname _A_d mxm.c
	tapenade -tangent -head "naive_mxm (B)\(C)" -o mxm_B.c $(TFLAGS) -tgtfuncname _B_d mxm.c
	tapenade -reverse -head "naive_mxm (A,B)\(C)" -o mxm.c $(TFLAGS) mxm.c
	tapenade -reverse -head "naive_mxm (A)\(C)" -o mxm_A.c $(TFLAGS) -adjfuncname _A_b mxm.c
	tapenade -reverse -head "naive_mxm (B)\(C)" -o mxm_B.c $(TFLAGS) -adjfuncname _B_b mxm.c

dgemm:
	tapenade -tangent -head "naive_dgemm (A B)\(C)" $(TFLAGS) mxm.c
	tapenade -tangent -head "naive_dgemm (A)\(C)" $(TFLAGS) -tgtfuncname _A_d mxm.c
	tapenade -tangent -head "naive_dgemm (B)\(C)" $(TFLAGS) -tgtfuncname _B_d mxm.c
	tapenade -reverse -head "naive_dgemm (A B)\(C)" $(TFLAGS) mxm.c
	tapenade -reverse -head "naive_dgemm (A)\(C)" $(TFLAGS) -adjfuncname _A_b mxm.c
	tapenade -reverse -head "naive_dgemm (B)\(C)" $(TFLAGS) -adjfuncname _B_b mxm.c

mtmv:
	tapenade -tangent -head "naive_mtmv (U)\(V)" $(TFLAGS) tensor.c
	tapenade -reverse -head "naive_mtmv (U)\(V)" $(TFLAGS) tensor.c

clean:
	rm -rf $(CLEANFILES)

cleandiff:
	rm -rf derivatives/*_d.c derivatives/*_d.msg derivatives/*_b.c derivatives/*_b.msg derivatives/*~

adBuffer.o: ${TAPENADE_HOME}/ADFirstAidKit/adBuffer.c
	gcc $(CFLAGS) -c -o $@ $^

adStack.o: ${TAPENADE_HOME}/ADFirstAidKit/adStack.c
	gcc $(CFLAGS) -c -o $@ $^

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
