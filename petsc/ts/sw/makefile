ALL: clean ex11
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex11 binaryoutput *.tap *.info *.txt *.o *.vtu

INCLUDES		= -I./
INCLUDES		+= -I${ADOLC_HOME}/ADOL-C/include
INCLUDES		+= -I${ADOLC_BUILDDIR}/ADOL-C/include

LIB 			= ${PETSC_TS_LIB}
LIB			+= -L${ADOLC_BUILDDIR}/lib64 -ladolc

CC 			= g++

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex11: ex11.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex11_sa: ex11_sa.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
