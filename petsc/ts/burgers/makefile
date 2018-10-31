ALL: clean burgers_findiff2d
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11 -I${ADOLC_BUILDDIR}/ADOL-C/include
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= burgers_findiff2d binaryoutput *.tap *.info *.txt *.o


LIB 			= ${PETSC_TS_LIB} -L${ADOLC_BUILDDIR}/lib64 -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

burgers_findiff2d: burgers_findiff2d.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
