ALL: petsc_version
CFLAGS          =
FFLAGS          =
CXXFLAGS        = -std=c++11 -I${ADOLC_HOME}/ADOL-C/include
CPPFLAGS        =
FPPFLAGS        =
CLEANFILES      = petsc_version reverse matfree *.tap *.o

INCLUDES	=
LIB             = ${PETSC_TS_LIB} -L${ADOLC_BUILDDIR}/lib64 -ladolc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#%.o:../../petsc/utils/%.cpp
#	g++ -o $@ $^ $(LIB)
#
#example_utils.o: example_utils.cpp
#	g++ -o $@ $^ $(LIB)
#
#../../petsc/utils/matfree.o: ../../petsc/utils/matfree.cpp
#	g++ -o $@ $^ $(LIB)

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

include ${PETSC_DIR}/lib/petsc/conf/test

