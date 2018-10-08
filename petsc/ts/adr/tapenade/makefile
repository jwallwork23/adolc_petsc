ALL: clean diff ex5
CFLAGS		= -c -g
FFLAGS		=
CPPFLAGS	=
FPPFLAGS	=
CLEANFILES	= *.msg *_d.c *_d.h *~ *.tap ex5

CC		= gcc
LIB		= ${PETSC_TS_LIB}
INCLUDES	= -I./

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#RHSLocal.o: RHSLocal.c ex5.h
#	gcc -o $@ $^ $(LIB)

#RHSLocal_d.o: RHSLocal_d.c
#	gcc -o $@ $^ $(LIB)

#Jacobian.o: Jacobian.c
#	gcc -o $@ $^ $(LIB)


diff: RHSLocal.o
	${CLINKER} $(INCLUDES) $(CFLAGS) -o $^ RHSLocal.c
	tapenade -tangent -head "RHSLocal ()\()" RHSLocal.c
	${CLINKER} $(INCLUDES) $(CFLAGS) -o RHSLocal_d.o RHSLocal_d.c
	${CLINKER} $(INCLUDES) $(CFLAGS) -o Jacobian.o Jacobian.c

ex5: ex5.o RHSLocal.o RHSLocal_d.o Jacobian.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} *.o

include ${PETSC_DIR}/lib/petsc/conf/test
