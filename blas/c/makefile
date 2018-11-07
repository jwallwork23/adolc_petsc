all: clean cleandiff dgemm test_AB

CFLAGS	= -Wall -I${TAPENADE_HOME}/ADFirstAidKit
CC	= gcc
TFLAGS  = -O derivatives

mpm:
	tapenade -tangent -head "mpm (A B)\(C)" -o mpm $(TFLAGS) mxm.c
	tapenade -tangent -head "mpm (A)\(C)" -o mpm_A $(TFLAGS) mxm.c
	tapenade -tangent -head "mpm (B)\(C)" -o mpm_B $(TFLAGS) mxm.c
	tapenade -reverse -head "mpm (A B)\(C)" -o mpm $(TFLAGS) mxm.c
	tapenade -reverse -head "mpm (A)\(C)" -o mpm_A $(TFLAGS) mxm.c
	tapenade -reverse -head "mpm (B)\(C)" -o mpm_B $(TFLAGS) mxm.c

mxm:
	tapenade -tangent -head "mxm (A B)\(C)" -o mxm.c $(TFLAGS) mxm.c
	tapenade -tangent -head "mxm (A)\(C)" -o mxm_A.c $(TFLAGS) mxm.c
	tapenade -tangent -head "mxm (B)\(C)" -o mxm_B.c $(TFLAGS) mxm.c
	tapenade -reverse -head "mxm (A,B)\(C)" -o mxm.c $(TFLAGS) mxm.c
	tapenade -reverse -head "mxm (A)\(C)" -o mxm_A.c $(TFLAGS) mxm.c
	tapenade -reverse -head "mxm (B)\(C)" -o mxm_B.c $(TFLAGS) mxm.c

dgemm:
	tapenade -tangent -head "dgemm (A B)\(C)" $(TFLAGS) mxm.c
	tapenade -tangent -head "dgemm (A)\(C)" $(TFLAGS) -tgtfuncname _A_d mxm.c
	tapenade -tangent -head "dgemm (B)\(C)" $(TFLAGS) -tgtfuncname _B_d mxm.c
	tapenade -reverse -head "dgemm (A B)\(C)" $(TFLAGS) mxm.c
	tapenade -reverse -head "dgemm (A)\(C)" $(TFLAGS) -adjfuncname _A_b mxm.c
	tapenade -reverse -head "dgemm (B)\(C)" $(TFLAGS) -adjfuncname _B_b mxm.c

clean:
	rm -rf test *.o *~

cleandiff:
	rm -rf derivatives/*_d.c derivatives/*_d.msg derivatives/*_b.c derivatives/*_b.msg

test_AB.o: test_AB.c
	$(CC) $(CFLAGS) -c -o $@ $^

test_AB: test_AB.o
	$(CC) -o $@ $^
	rm *.o

test_A.o: test_A.c
	$(CC) $(CFLAGS) -c -o $@ $^

test_A: test_A.o
	$(CC) -o $@ $^
	rm *.o
