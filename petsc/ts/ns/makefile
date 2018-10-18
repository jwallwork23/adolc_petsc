ALL: clean diff ex46
CFLAGS			= -Wall
FFLAGS			=
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex46 binaryoutput *.tap *.info *.txt *.o *.vtu *.msg *_d.c *_d.h *~

LIB 			= ${PETSC_TS_LIB}

CC 			= gcc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

diff:
	tapenade -tangent -head "f0_mms1_u (u u_t)\(f0)" residuals.c

ex46: ex46.o
	$(CC) $(INCLUDES) $(CFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
