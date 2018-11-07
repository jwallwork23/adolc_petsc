all: clean cleandiff dgemm_forward dgemm_reverse

CFLAGS	= -Wall -I${TAPENADE_HOME}/ADFirstAidKit
CC	= gcc

mpm:
	tapenade -tangent -head "mpm (A B)\(C)" -o mpm mxm.c
	tapenade -reverse -head "mpm (A B)\(C)" -o mpm mxm.c

mxm:
	tapenade -tangent -head "mxm (A B)\(C)" mxm.c
	tapenade -reverse -head "mxm (A,B)\(C)" mxm.c

dgemm:
	tapenade -tangent -head "dgemm (A B)\(C)" -o dgemm mxm.c
	tapenade -reverse -head "dgemm (A B)\(C)" -o dgemm mxm.c

clean:
	rm -rf test *.o *~

cleandiff:
	rm -rf *_d.c *_d.msg *_b.c *_b.msg

test.o: test.c
	$(CC) $(CFLAGS) -c -o $@ $^

test: test.o
	$(CC) -o $@ $^
	rm *.o
