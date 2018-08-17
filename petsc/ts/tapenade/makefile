ALL: clean diff ex16adj
CFLAGS		=
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= *.msg *_d.c *_d.h *~ *.tap ex16adj ex16opt_p ex16opt_ic

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

diff: rhs.o
	gcc -c -o $^ rhs.c
	tapenade -tangent -head "rhs (x)\(f)" -tgtfuncname "_d" -tgtvarname "_d" rhs.c
	gcc -c -o rhs_d.o rhs_d.c
	tapenade -tangent -head "rhs (mu)\(f)" -tgtfuncname "p_d" -tgtvarname "p_d" rhs.c
	gcc -c -o rhsp_d.o rhsp_d.c
	gcc -c -o jacobian.o jacobian.c

ex16adj: ex16adj.o rhs.o rhs_d.o rhsp_d.o jacobian.o
	-${CLINKER} -o $@ $^  ${PETSC_TS_LIB}
	${RM} *.o

# ex16adj: ex16adj.o rhs.o rhs_d.o rhsp_d.o jacobian.o chkopts
	# -${CLINKER} -o $@ ex16adj.o rhs.o rhs_d.o rhsp_d.o jacobian.o  ${PETSC_TS_LIB}
	# ${RM} *.o

ex16opt_p: ex16opt_p.o rhs.o rhs_d.o rhsp_d.o jacobian.o
	-${CLINKER} -o $@ $^  ${PETSC_TS_LIB}
	${RM} *.o

ex16opt_ic: ex16opt_ic.o rhs.o rhs_d.o rhsp_d.o jacobian.o
	-${CLINKER} -o $@ $^  ${PETSC_TS_LIB}
	${RM} *.o

include ${PETSC_DIR}/lib/petsc/conf/test
