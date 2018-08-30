ALL: clean diff ex5
CFLAGS		=
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= *.msg *_d.c *_d.h *~ *.tap ex5

# Compiler
COMPILE		= gcc	# gnu

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

diff: RHSLocal.o
	$(COMPILE) -c -o $^ RHSLocal.o
	tapenade -tangent -head "RHSLocal ()\()" RHSLocal.c
	$(COMPILE) -c -o RHSLocal_d.o RHSLocal_d.c
	$(COMPILE) -c -o Jacobian.o Jacobian.c

ex5: ex5.o RHSLocal.o RHSLocal_d.o Jacobian.o
	-${CLINKER} -o $@ $^ ${PETSC_TS_LIB}
	${RM} *.o

include ${PETSC_DIR}/lib/petsc/conf/test
