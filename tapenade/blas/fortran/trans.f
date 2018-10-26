*  =================================================================
*
      SUBROUTINE TRANSPOSE(M,N,A,LDA,UDA,C)
*
*     Scalar arguments
      INTEGER K,M,N,LDA,UDA
*
*     Array arguments
      DOUBLE PRECISION A(LDA,UDA),C(UDA,LDA)
*
*  =================================================================
*
*     Local scalars
      DOUBLE PRECISION TEMP
      INTEGER I,J
*
*     C := A**T
*
      DO 10 J = 1,N
          DO 20 I = 1,M
              TEMP = A(I,J)
              C(J,I) = TEMP
   20     CONTINUE
   10 CONTINUE
*
      RETURN
*
      END
