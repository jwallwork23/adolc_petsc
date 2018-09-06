ALL: ex1
CFLAGS		=
FFLAGS		=
CXXFLAGS	= ${ADOLC_HOME}/build/include
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex1 *.tap

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o
	g++ -std=c++11 -o $@ $^ ${PETSC_SNES_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
