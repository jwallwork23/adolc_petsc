ALL: clean ex16adj
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/include
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex16adj ex16opt_ic ex20adj *.tap *.txt *.o

LIB		= ${PETSC_TS_LIB} -L${USER_LIB} -lboost_system
LIB             += -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
LIB             += -L${COLPACK_HOME}/lib -lColPack -Wl,-rpath,${COLPACK_HOME}/lib -lm

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex16adj: ex16adj.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex16opt_ic: ex16opt_ic.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex20adj: ex20adj.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
