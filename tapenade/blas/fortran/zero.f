*  =================================================================
*
      SUBROUTINE ZEROOUT(M,N,A,LDA)
*
*     Scalar arguments
      INTEGER M,N,LDA
*
*     Array arguments
      DOUBLE PRECISION A(LDA,*)
*
*  =================================================================
*
*     Local scalars
      DOUBLE PRECISION TEMP
      INTEGER I,J
*
*     Parameters
      PARAMETER ZERO=0.0D+0
*
*     A := 0
*
      DO 10 J = 1,N
          DO 20 L = 1,K
              A(I,J) = ZERO
   20     CONTINUE
   10 CONTINUE
*
      RETURN
*
      END
