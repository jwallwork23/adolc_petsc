ALL: clean ex16adj ex16opt_ic ex20adj
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex16adj ex16opt_ic ex20adj *.tap
CC		= g++

INCLUDES	=
LIB		= ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex16adj: ex16adj.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex16opt_ic: ex16opt_ic.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex20adj: ex20adj.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
