ALL: test
CFLAGS		= ${ADOLC_HOME}/build/include
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= test *.tap

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

test: test.o
	g++ -std=c++11 -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
