ALL: clean ex1 ex5 ex5adj
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex1 ex5 ex5adj binaryoutput *.tap *.info

INCLUDES		= -I${ADOLC_HOME}/include -I${COLPACK_HOME}/include
LIB 			= ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc

# Flags for ColPack TODO: link this properly - the below is currently ineffective
COLPACK_INSTALL_PATH 	= ${COLPACK_HOME}/libs/libColPack.so
# LDFLAGS 		= -Wall -fopenmp -O3 -ldl ${COLPACK_INSTALL_PATH}

CC 			= g++

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o
	$(CC) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIB)
	${RM} $^

ex5: ex5.o
	$(CC) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LD_FLAGS) $(LIB)
	${RM} $^

ex5adj: ex5adj.o
	$(CC) $(CXXFLAGS) $(INCLUDES) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
