all: clean exec


exec:
	gcc -c -o f1.o f1.c
	tapenade -tangent -head "f1 (xx)\(ff)" f1.c
	gcc -c -o f1_d.o f1_d.c
	gcc -c -o jacobian.o jacobian.c
	gcc -c -o ex1.o ex1.c
	gcc -o exec f1.o f1_d.o jacobian.o ex1.o
	rm *.o

clean:
	rm -Rf *.msg *_d.c *_d.h *~ *.tap exec
