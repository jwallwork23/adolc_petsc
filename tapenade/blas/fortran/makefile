all: clean cleandiff forward reverse

forward:
	tapenade -tangent -head "DGEMM (A B)\(C)" dgemm.f

reverse:
	tapenade -reverse -head "DGEMM (A B)\(C)" dgemm.f

clean:
	rm -rf *.o

cleandiff:
	rm -rf *_d.f *_b.f *.msg
