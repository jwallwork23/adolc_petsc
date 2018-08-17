ALL: clean diff ex16adj
CFLAGS		=
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= *.msg *_d.c *_d.h *~ *.tap ex16adj

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

diff: rhs.o
	gcc -c -o rhs.o rhs.c
	tapenade -tangent -head "rhs (x)\(f)" rhs.c
	gcc -c -o rhs_d.o rhs_d.c
	gcc -c -o jacobian.o jacobian.c

ex16adj: ex16adj.o rhs.o rhs_d.o jacobian.o chkopts
	-${CLINKER} -o ex16adj ex16adj.o rhs.o rhs_d.o jacobian.o  ${PETSC_TS_LIB}
	${RM} ex16.o

include ${PETSC_DIR}/lib/petsc/conf/test
