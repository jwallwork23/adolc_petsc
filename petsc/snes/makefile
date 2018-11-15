ALL: ex1
CFLAGS		=
FFLAGS		=
CXXFLAGS	= -std=c++11 -I${ADOLC_BUILDDIR}/include
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex1 *.tap

LIB		= ${PETSC_SNES_LIB} -L${USER_LIB} -lboost_system
LIB		+= -L${ADOLC_BUILDDIR}/lib64 -ladolc -Wl,-rpath,${ADOLC_BUILDDIR}/lib64
LIB             += -L${COLPACK_HOME}/lib -lColPack -Wl,-rpath,${COLPACK_HOME}/lib -lm

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

ex1: ex1.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
