ALL: clean ex11
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11 -I${ADOLC_BUILDDIR}/ADOL-C/include
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex11 binaryoutput *.tap *.info *.txt *.o *.vtu

LIB 			= ${PETSC_TS_LIB} -L${ADOLC_BUILDDIR}/lib64 -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex11: ex11.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex11_sa: ex11_sa.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
