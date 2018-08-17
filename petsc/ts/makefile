ALL: ex16adj
CFLAGS		= ${ADOLC_HOME}/build/include
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex16adj *.tap

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex16adj: ex16adj.o chkopts
	g++ -std=c++11 -o ex16adj ex16adj.o ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} ex16adj.o

include ${PETSC_DIR}/lib/petsc/conf/test
