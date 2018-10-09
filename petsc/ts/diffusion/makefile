ALL: clean ex13
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex13 ex13hong *.tap *.txt *.o
CC		= g++

INCLUDES	=
LIB		= ${PETSC_TS_LIB} -L${ADOLC_BUILDDIR} -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

%.o: ../utils/%.cpp
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)

ex13: ex13.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex13hong: ex13hong.o
	${CLINKER} -o $@ $^ ${PETSC_TS_LIB}
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
