ALL: ex16adj
CFLAGS		= ${ADOLC_HOME}/build/include
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex16adj ex16opt_ic *.tap

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex16adj: ex16adj.o
	g++ -std=c++11 -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

ex16opt_ic: ex16opt_ic.o
	g++ -std=c++11 -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
