ALL: clean ex5imp
CFLAGS			=
FFLAGS			=
CXXFLAGS		=
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex5imp

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex5imp: ex5imp.o
	${CLINKER} -o $@ $^ ${PETSC_TS_LIB}
	${RM} $^


include ${PETSC_DIR}/lib/petsc/conf/test
