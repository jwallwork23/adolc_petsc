ALL: ex5adj
CFLAGS			= -std=c++11 ${ADOLC_HOME}/build/include ${COLPACK_HOME}/include
FFLAGS			=
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex1 ex4 ex5 ex5adj test binaryoutput *.tap *.info

# Flags for ColPack
COLPACK_INSTALL_PATH 	= ${COLPACK_HOME}/libs/libColPack.so
LDFLAGS 		= -Wall -fopenmp -O3 -std=c++11 -ldl ${COLPACK_INSTALL_PATH}

# Compiler
COMPILE 		= g++	# gnu

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o
	$(COMPILE) $(CCFLAGS) -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

ex4: ex4.o
	$(COMPILE) $(CCFLAGS) -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

ex5: ex5.o
	$(COMPILE) $(CCFLAGS) -o $@ $^ ${PETSC_TS_LIB} $(LD_FLAGS) -L${ADOLC_HOME}/build -ladolc
	${RM} $^

ex5adj: ex5adj.o
	$(COMPILE) $(CCFLAGS) -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

test: test.o
	$(COMPILE) $(CCFLAGS) -o $@ $^ ${PETSC_TS_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
