all: clean cleandiff forward reverse

forward:
	tapenade -tangent -head "DGEMM (A B)\(C)" ${LAPACK_HOME}/BLAS/SRC/dgemm.f

reverse:
	tapenade -reverse -head "DGEMM (A B)\(C)" ${LAPACK_HOME}/BLAS/SRC/dgemm.f

test.o: test.f
	f77 -Wall -c -o $@ $^

test: test.o
	f77 -Wall -o $@ $^ -L${LAPACK_HOME} -llapack -lblas

clean:
	rm -rf *.o test

cleandiff:
	rm -rf *_d.f *_b.f *.msg
