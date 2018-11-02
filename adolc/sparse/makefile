ALL: petsc_version
FFLAGS          =
CFLAGS		=
CXXFLAGS        = -std=c++11 -I${ADOLC_BUILDDIR}/include
CPPFLAGS	=
FPPFLAGS        =
CLEANFILES      = petsc_version reverse matfree traceless *.tap *.o exec
CC              = g++

LIB             = ${PETSC_TS_LIB} -L${USER_LIB} -lboost_system
LIB		+= -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
LIB		+= -L${COLPACK_HOME}/lib -lColPack -Wl,-rpath,${COLPACK_HOME}/lib -lm

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

petsc_version: petsc_version.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

reverse: reverse.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

matfree: matfree.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

traceless: traceless.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

test: test.o
	g++ -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test

