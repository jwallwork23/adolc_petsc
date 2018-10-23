ALL: clean ex47
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11 -I${ADOLC_BUILDDIR}/include
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex47 binaryoutput *.tap *.info *.txt *.o *.vtu

LIB 			= ${PETSC_TS_LIB} -L${ADOLC_BUILDDIR}/lib64 -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex47: ex47.o
	-${CLINKER}-o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
