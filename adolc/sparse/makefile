ALL: petsc_version
CFLAGS          =
FFLAGS          =
CXXFLAGS        = -std=c++11
CPPFLAGS        =
FPPFLAGS        =
CLEANFILES      = petsc_version reverse *.tap
CC              = g++
LIB             = ${PETSC_SNES_LIB} -L${ADOLC_HOME}/build -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

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

