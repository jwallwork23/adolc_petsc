ALL: ex5adj
CFLAGS		= ${ADOLC_HOME}/build/include
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex5 ex5adj *.tap

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex5: ex5.o
	g++ -std=c++11 -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

ex5adj: ex5adj.o
	g++ -std=c++11 -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
