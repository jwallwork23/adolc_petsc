ALL: clean ex9bus
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/include
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex9bus ex9busadj *.tap *.info *.o *.txt

LIB 		= ${PETSC_TS_LIB} -L${USER_LIB} -lboost_system
LIB             += -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
LIB             += -L${COLPACK_HOME}/lib -lColPack -Wl,-rpath,${COLPACK_HOME}/lib -lm

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex9bus: ex9bus.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

ex9busadj: ex9busadj.o
	$(CC) $(INCLUDES) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
