ALL: clean diff ex46
CFLAGS			=
FFLAGS			=
CPPFLAGS		=
FPPFLAGS		=
CLEANFILES		= ex46 binaryoutput *.tap *.info *.txt utils/*.msg utils/*_d.c utils/*~ utils/*_dv.c

LIB 			= ${PETSC_TS_LIB}

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#          g0_uu                  g1_uu                g1_pu
DERIV	:= f0_mms1_u (u u_t)\(f0) f0_mms2_u (u_x)\(f0) f0_p (u_x)\(f0)
#                    g2_up         g3_uu
DERIV	:= "$(DERIV) f1_u (u)\(f1) f1_u (u_x)(f1)"

diff:
	tapenade -tangent -head $(DERIV) -O utils utils/residuals.c

diff2:
	tapenade -multi -head $(DERIV) -O utils utils/residuals.c

ex46: ex46.o
	-${CLINKER} -o $@ $^ $(LIB)
	${RM} $^

include ${PETSC_DIR}/lib/petsc/conf/test
