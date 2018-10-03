ALL: petsc_version
CFLAGS          =
FFLAGS          =
CXXFLAGS        = -std=c++11
CPPFLAGS        =
FPPFLAGS        =
CLEANFILES      = petsc_version reverse matfree *.tap *.o
CC              = g++

INCLUDES	=
LIB             = ${PETSC_TS_LIB} -L${ADOLC_BUILDDIR} -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#%.o:../../petsc/utils/%.cpp
#	$(CC) $(CXXFLAGS) -o $@ $^ $(LIB)

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

