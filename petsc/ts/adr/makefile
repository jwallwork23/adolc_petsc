ALL: clean ex5
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex1 ex5 ex5adj ex5imp ex5mf binaryoutput *.tap *.info *.txt *.o

INCLUDES		= -I./
INCLUDES		+= -I${ADOLC_HOME}/ADOL-C/include
INCLUDES		+= -I${ADOLC_BUILDDIR}/ADOL-C/include

LIB 			= ${PETSC_TS_LIB}
LIB			+= -L${ADOLC_HOME}/build -ladolc

CC 			= g++

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#sparse.o: ../../utils/sparse.cpp
#	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)

ex1: ex1.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex5: ex5.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex5mf: ex5mf.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex5adj: ex5adj.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex5imp: ex5imp.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^


include ${PETSC_DIR}/lib/petsc/conf/test
