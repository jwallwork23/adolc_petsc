CFLAGS		=
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

all: diff run

diff: f1.o
	gcc -c -o f1.o f1.c
	tapenade -tangent -head "f1 (xx)\(ff)" f1.c
	gcc -c -o f1_d.o f1_d.c
	gcc -c -o jacobian.o jacobian.c

run: run.o f1.o f1_d.o jacobian.o chkopts
	-${CLINKER} -o run run.o f1.o f1_d.o jacobian.o ${PETSC_SNES_LIB}
	${RM} *.o

cleanup:
	rm -Rf *.msg *_d.c *_d.h *~ *.tap run


include ${PETSC_DIR}/lib/petsc/conf/test
