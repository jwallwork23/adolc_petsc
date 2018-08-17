ALL: clean diff exec
CFLAGS		=
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= *.msg *_d.c *_d.h *~ *.tap exec

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

diff: f1.o f2.o
	gcc -c -o f1.o f1.c
	gcc -c -o f2.o f2.c
	tapenade -tangent -head "f1 (xx)\(ff)" f1.c
	tapenade -tangent -head "f2 (xx)\(ff)" f2.c
	gcc -c -o f1_d.o f1_d.c
	gcc -c -o f2_d.o f2_d.c
	gcc -c -o jacobian.o jacobian.c

exec: run.o f1.o f2.o f2_d.o f1_d.o jacobian.o
	-${CLINKER} -o $@ $^ ${PETSC_SNES_LIB}
	${RM} *.o

# exec: run.o f1.o f2.o f2_d.o f1_d.o jacobian.o chkopts
# 	-${CLINKER} -o exec run.o f1.o f2.o f1_d.o f2_d.o jacobian.o ${PETSC_SNES_LIB}
# 	${RM} *.o

include ${PETSC_DIR}/lib/petsc/conf/test
