ALL: ex1
CFLAGS		= ${ADOLC_HOME}/build/include
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= ex1

# HOW TO LINK?: -L${ADOLC_HOME}/build -ladolc
# EXAMPLESC	= ex1.cpp

SRC=ex1.cpp
OBJ=ex1.o

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

# %.o:%.cpp
# 	g++ -std=c++11 -I${ADOLC_HOME}/build/include $^ -c -o$@

ex1: $(OBJ)
	g++ -std=c++11 -o ex1 $(OBJ) ${PETSC_SNES_LIB} -L${ADOLC_HOME}/build -ladolc
	${RM} ex1.o

cleanup:
	${RM} ex1 *.tap

# ex1: $(OBJ) chkopts
# 	-${CLINKER} -std=c++11 -o ex1 $(OBJ) ${PETSC_SNES_LIB}
# 	${RM} ex1.o

# include ${PETSC_DIR}/lib/petsc/conf/test
