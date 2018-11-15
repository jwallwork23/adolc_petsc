ALL: clean ex5_mf
CFLAGS			=
FFLAGS			=
CXXFLAGS		=
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex5_mf ex5adj_mf binaryoutput *.tap *.info *.txt *.o

INCLUDES		= -I./

LIB 			= ${PETSC_TS_LIB}

CC 			= ${CLINKER}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex5_mf: ex5_mf.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex5adj_mf: ex5adj_mf.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
