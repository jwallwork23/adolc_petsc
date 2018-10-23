ALL: clean ex46
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11 -I${ADOLC_BUILDDIR}/ADOL-C/include
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex46 binaryoutput *.tap *.info *.txt *.o

LIB 			= ${PETSC_TS_LIB} -L${ADOLC_BUILDDIR}/lib64 -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex46: ex46.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
