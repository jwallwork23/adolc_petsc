ALL: clean ex13
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex13 *.tap *.txt *.o
CC		= g++

INCLUDES	=
LIB		= ${PETSC_TS_LIB} -L${ADOLC_BUILDDIR} -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex13: ex13.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
