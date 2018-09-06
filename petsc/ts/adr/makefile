ALL: ex5adj
CFLAGS			=
FFLAGS			=
CXXFLAGS		= -std=c++11 ${ADOLC_HOME}/build/include ${COLPACK_HOME}/include
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex1 ex5 ex5adj binaryoutput *.tap *.info

# Flags for ColPack TODO: link this properly - the below is currently ineffective
COLPACK_INSTALL_PATH 	= ${COLPACK_HOME}/libs/libColPack.so
LDFLAGS 		= -Wall -fopenmp -O3 -std=c++11 -ldl ${COLPACK_INSTALL_PATH}

# Compiler
COMPILE 		= g++	# gnu

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o
	$(COMPILE) $(CCFLAGS) -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

ex5: ex5.o
	$(COMPILE) $(CCFLAGS) -o $@ $^ ${PETSC_TS_LIB} $(LD_FLAGS) -L${ADOLC_HOME}/build -ladolc
	${RM} $^

ex5adj: ex5adj.o
	$(COMPILE) $(CCFLAGS) -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
