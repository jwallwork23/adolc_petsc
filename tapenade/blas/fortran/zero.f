C  =====================================================================
C
      SUBROUTINE ZEROOUT(M,N,A,LDA)
C
C     .. Scalar arguments ..
      INTEGER M,N,LDA
C
C     .. Array arguments ..
      DOUBLE PRECISION A(LDA,*)
C
C  =====================================================================
C
C     .. Local scalars ..
      INTEGER I,J
C     ..
C     .. Parameters ..
      DOUBLE PRECISION ZERO
      PARAMETER ZERO=0.0D+0
C
C     A := 0
C
      DO 10 J = 1,N
          DO 20 I = 1,M
              A(I,J) = ZERO
   20     CONTINUE
   10 CONTINUE
C
      RETURN
C
      END
