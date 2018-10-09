ALL: clean diff diff_p ex16adj
CFLAGS		=
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= *.msg *_d.c *_d.h *~ *.tap ex16adj ex16opt_p ex16opt_ic

# Compiler
CC		= gcc	# gnu

DEPS		= rhs.o rhs_d.o jacobian.o
DEPS_P		= rhsp_d.o jacobianp.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

rhs.o: rhs.c
	$(CC) -c $^

rhs_d.o: rhs_d.c
	$(CC) -c $^

rhsp_d.o: rhsp_d.c
	$(CC) -c $^

jacobian.o: jacobian.c
	$(CC) -c $^

jacobianp.o: jacobianp.c
	$(CC) -c $^

diff: rhs.o
	tapenade -tangent -head "rhs (x)\(f)" -tgtfuncname "_d" -tgtvarname "_d" rhs.c

diff_p: rhs.o
	tapenade -tangent -head "rhs (mu)\(f)" -tgtfuncname "p_d" -tgtvarname "p_d" rhs.c

ex16adj: ex16adj.o $(DEPS) $(DEPS_P)
	-${CLINKER} -o $@ $^ ${PETSC_TS_LIB}
	${RM} *.o

ex16opt_p: ex16opt_p.o $(DEPS) $(DEPS_P)
	-${CLINKER} -o $@ $^ ${PETSC_TS_LIB}
	${RM} *.o

ex16opt_ic: ex16opt_ic.o $(DEPS)
	-${CLINKER} -o $@ $^ ${PETSC_TS_LIB}
	${RM} *.o

include ${PETSC_DIR}/lib/petsc/conf/test
