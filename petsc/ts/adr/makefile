ALL: clean ex1 ex5 ex5adj
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex1 ex5 ex5adj binaryoutput *.tap *.info

INCLUDES	= -I./
INCLUDES	+= -I${ADOLC_HOME}/ADOL-C/include
INCLUDES	+= -I${ADOLC_BUILDDIR}/ADOL-C/include
INCLUDES	+= -I${COLPACK_HOME}/include/ColPack
INCLUDES        += -I${COLPACK_HOME}/Main
INCLUDES        += -I${COLPACK_HOME}/GraphColoring
INCLUDES        += -I${COLPACK_HOME}/BipartiteGraphColoring
INCLUDES        += -I${COLPACK_HOME}/BipartiteGraphPartialColoring
INCLUDES        += -I${COLPACK_HOME}/Utilities
INCLUDES        += -I${COLPACK_HOME}/Recovery

LIB 		= ${PETSC_TS_LIB}
LIB		+= -L${ADOLC_HOME}/build -ladolc
#LIB		+= -ldl ${COLPACK_INSTALL_PATH}

# TODO: link ColPack properly - the below is currently ineffective
COLPACK_INSTALL_PATH 	= ${COLPACK_HOME}/lib/libColPack.so
OMP_FLAG		= -fopenmp
LDFLAGS 		= -Wall -std=c++11 $(OMP_FLAG) -O3 -ldl ${COLPACK_INSTALL_PATH}

CC 			= g++

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex5: ex5.o
	$(CC) $(INCLUDES) $(LD_FLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex5adj: ex5adj.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
