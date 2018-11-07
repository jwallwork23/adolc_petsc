all: clean cleandiff dgemm test

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
	tapenade -tangent -head "dgemm (A B)\(C)" -o dgemm $(TFLAGS) mxm.c
	tapenade -tangent -head "dgemm (A)\(C)" -o dgemm_A $(TFLAGS) mxm.c
	tapenade -tangent -head "dgemm (B)\(C)" -o dgemm_B $(TFLAGS) mxm.c
	tapenade -reverse -head "dgemm (A B)\(C)" -o dgemm $(TFLAGS) mxm.c
	tapenade -reverse -head "dgemm (A)\(C)" -o dgemm_A $(TFLAGS) mxm.c
	tapenade -reverse -head "dgemm (B)\(C)" -o dgemm_B $(TFLAGS) mxm.c

clean:
	rm -rf test *.o *~

cleandiff:
	rm -rf derivatives/*_d.c derivatives/*_d.msg derivatives/*_b.c derivatives/*_b.msg

test.o: test.c
	$(CC) $(CFLAGS) -c -o $@ $^

test: test.o
	$(CC) -o $@ $^
	rm *.o
