all: clean cleandiff forward reverse

CFLAGS	= -Wall
CC	= gcc


forward:
	rm -rf *_d.c *_d.msg
	tapenade -tangent -head "mxm (A B)\(C)" mxm.c

reverse:
	rm -rf *_b.c *_b.msg
	tapenade -reverse -head "mxm (A B)\(C)" mxm.c

clean:
	rm -rf test *.o

cleandiff:
	rm -rf *_d.c *_d.msg *_b.c *_b.msg

test.o: test.c
	$(CC) $(CFLAGS) -c -o $@ $^


test: test.o
	$(CC) -o $@ $^
	rm *.o
