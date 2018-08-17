ALL: ex1
CFLAGS		= ${ADOLC_HOME}/build/include
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex1 *.tap

SRC=ex1.cpp
OBJ=ex1.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: $(OBJ) chkopts
	g++ -std=c++11 -o $@ $(OBJ) ${PETSC_SNES_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $(OBJ)

include ${PETSC_DIR}/lib/petsc/conf/test
