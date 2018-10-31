ALL: clean diff ex5adj_mf
CFLAGS		=
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex5adj_mf *.msg *_d.c *_d.h *~ *.tap ex5

INCLUDES	= -I./
LIB		= ${PETSC_TS_LIB}
CC		= ${CLINKER}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

diff: RHSLocal.o
#	${CLINKER} $(INCLUDES) $(CFLAGS) -o $^ RHSLocal.c
	tapenade -tangent -head "RHSLocal ()\()" RHSLocal.c
#	${CLINKER} $(INCLUDES) $(CFLAGS) -o RHSLocal_d.o RHSLocal_d.c
#	${CLINKER} $(INCLUDES) $(CFLAGS) -o Jacobian.o Jacobian.c

ex5adj_mf: ex5adj_mf.o
	$(CC) $(INCLUDES) $(CFLAGS) -o $@ $^ $(LIB)
	${RM} *.o

include ${PETSC_DIR}/lib/petsc/conf/test
