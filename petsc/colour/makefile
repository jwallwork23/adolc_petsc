ALL: ex
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex *.tap
CC		= g++
LIB		= ${PETSC_SNES_LIB}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex: ex.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
