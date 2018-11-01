ALL: petsc_version
CFLAGS          =
FFLAGS          =
CXXFLAGS        = -std=c++11 -I${ADOLC_BUILDDIR}/include
CPPFLAGS        =
FPPFLAGS        =
CLEANFILES      = petsc_version reverse matfree *.tap *.o exec
CC              = g++

LIB             = ${PETSC_TS_LIB} -L${USER_LIB} -lboost_system
LIB		+= -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
LIB		+= -L${COLPACK_HOME}/lib -lColPack -Wl,-rpath,${COLPACK_HOME}/lib -lm

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

example_utils.o: example_utils.cpp
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)

../../petsc/utils/matfree.o: ../../petsc/utils/matfree.cpp
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)

petsc_version: petsc_version.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

reverse: reverse.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

matfree: matfree.o
	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test

