ALL: ex1
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/ADOL-C/include
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex1 *.tap
CC		= g++
LIB		= ${PETSC_SNES_LIB} -L${ADOLC_HOME}/build -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
