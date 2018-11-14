all: clean cleandiff mtmv test_tensor

CFLAGS		= -Wall -I${TAPENADE_HOME}/ADFirstAidKit -I${LAPACK_HOME}/CBLAS
LIB		= -L${LAPACK_HOME} -lcblas -lrefblas -ltmglib -llapack
TFLAGS  	= -O derivatives
CLEANFILES	= test_AB test_A test_tensor *.o *~

derivatives/mxm_d.c: mxm.c
	tapenade -d -head "naive_mxm[AB](A B)\(C)   \
                           naive_mxm[A](A)\(C)      \
                           naive_mxm[B](B)\(C)      \
                           naive_mpm[AB](A B)\(C)   \
                           naive_mpm[A](A)\(C)      \
                           naive_mpm[B](B)\(C)      \
                           naive_dgemm[AB](A B)\(C) \
                           naive_dgemm[A](A)\(C)    \
                           naive_dgemm[B](B)\(C)    \
                           naive_mtmv(U)\(V)        \
                           " mxm.c $(TFLAGS)

derivatives/mxm_b.c: mxm.c
	tapenade -b -head "naive_mxm[AB](A B)\(C)   \
                           naive_mxm[A](A)\(C)      \
                           naive_mxm[B](B)\(C)      \
                           naive_mpm[AB](A B)\(C)   \
                           naive_mpm[A](A)\(C)      \
                           naive_mpm[B](B)\(C)      \
                           naive_dgemm[AB](A B)\(C) \
                           naive_dgemm[A](A)\(C)    \
                           naive_dgemm[B](B)\(C)    \
                           naive_mtmv(U)\(V)        \
                           " mxm.c $(TFLAGS)

verify_mxm:
	civl compare -impl -DSIZELIMIT=5 verification_drivers/driver_mxm_dot.c -spec -DSIZELIMIT=5 verification_drivers/driver_mxm_dAB.c

verify_mpm:
	civl compare -impl -DSIZELIMIT=5 verification_drivers/driver_mpm_dot.c -spec -DSIZELIMIT=5 verification_drivers/driver_mpm_dAB.c

verify_dgemm:
	civl compare -impl -DSIZELIMIT=5 verification_drivers/driver_dgemm_dot.c -spec -DSIZELIMIT=5 verification_drivers/driver_dgemm_dAB.c
	civl compare -impl -DSIZELIMIT=5 verification_drivers/driver_dgemm_bar.c -spec -DSIZELIMIT=5 verification_drivers/driver_dgemm_bAB.c

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
