ALL: clean ex16adj
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11 ${ADOLC_HOME}/build/include
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex16adj ex16opt_ic ex20adj *.tap

# Compiler
COMPILE		= gcc	# gnu

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex16adj: ex16adj.o
	$(COMPILE) -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

ex16opt_ic: ex16opt_ic.o
	$(COMPILE) -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

ex20adj: ex20adj.o
	$(COMPILE) -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
