CFLAGS		=
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules


run: run.o chkopts
	gcc -c -o f1.o f1.c
	tapenade -tangent -head "f1 (xx)\(ff)" f1.c
	gcc -c -o f1_d.o f1_d.c
	-${CLINKER} -o run run.o f1.o f1_d.o ${PETSC_SNES_LIB}
	${RM} run.o f1.o f1_d.o


include ${PETSC_DIR}/lib/petsc/conf/test
