ALL: clean diff ex5
CFLAGS		=
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= *.msg *_d.c *_d.h *~ *.tap ex5

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

diff: rhs.o
	gcc -c -o $^ rhs.o

ex5: ex5.o rhs.o
	-${CLINKER} -o $@ $^ ${PETSC_TS_LIB}
	${RM} *.o

include ${PETSC_DIR}/lib/petsc/conf/test
