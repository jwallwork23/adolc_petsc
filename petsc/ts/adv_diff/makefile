ALL: clean ex47
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex47 binaryoutput *.tap *.info *.txt *.o *.vtu

INCLUDES		= -I./
INCLUDES		+= -I${ADOLC_HOME}/ADOL-C/include
INCLUDES		+= -I${ADOLC_BUILDDIR}/ADOL-C/include

LIB 			= ${PETSC_TS_LIB}
LIB			+= -L${ADOLC_HOME}/build -ladolc

CC 			= g++

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex47: ex47.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
