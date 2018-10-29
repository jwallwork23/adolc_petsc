all: clean cleandiff forward reverse

forward:
	tapenade -tangent -head "DGEMM (A B)\(C)" dgemm.f

reverse:
	tapenade -reverse -head "DGEMM (A B)\(C)" dgemm.f

test.o: test.f
	gfortran -c -o $@ $^

test: test.o
	gfortran -o $@ $^

clean:
	rm -rf *.o test

cleandiff:
	rm -rf *_d.f *_b.f *.msg
