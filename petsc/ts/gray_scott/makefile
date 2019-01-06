ALL: clean ex5
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/include
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex5 ex5adj ex5imp ex5mf binaryoutput *.tap *.info *.txt *.o

# No longer needed since --download-adolc and --download-colpack options can now be passed to petsc configure
LIB		= ${PETSC_TS_LIB} -L${USER_LIB} -lboost_system
LIB             += -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
LIB             += -L${COLPACK_HOME}/lib -lColPack -Wl,-rpath,${COLPACK_HOME}/lib -lm

#LIB		= ${PETSC_TS_LIB}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex5: ex5.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex5mf: ex5mf.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex5adj: ex5adj.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

ex5imp: ex5imp.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^


include ${PETSC_DIR}/lib/petsc/conf/test
