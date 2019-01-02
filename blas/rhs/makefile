all: clean forward reverse test
CFLAGS = -O3 -Wall -I. -I${TAPENADE_HOME}/ADFirstAidKit

cleandiff:
	rm -rf *.msg *_d.* *_b.* *~
cleanobj:
	rm -rf *.o test
clean: cleandiff cleanobj


### TAPENADE UTILS

adBuffer.o: ${TAPENADE_HOME}/ADFirstAidKit/adBuffer.c
	gcc $(CFLAGS) -c -o $@ $^

adStack.o: ${TAPENADE_HOME}/ADFirstAidKit/adStack.c
	gcc $(CFLAGS) -c -o $@ $^

### DIFFERENTIATION OF BURGERS' EQUATION RHS IN DGEMM FORMAT

forward: mxm.c
	tapenade -d -head "RHS(ul)\(outl)" rhs.c
	cp mxm.h mxm_d.h
	python postproc.py

reverse: mxm.c
	tapenade -b -head "RHS(ul)\(outl)" rhs.c
	cp mxm.h mxm_b.h
	python postproc.py -b 1

rhs_test.o: rhs_test.c
	gcc -c rhs_test.c

test: rhs_test.o adBuffer.o adStack.o
	gcc rhs_test.o -o test
