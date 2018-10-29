C  =====================================================================
C
      SUBROUTINE ZEROUT(M,N,A)
C
C     .. Scalar arguments ..
      INTEGER M,N
C
C     .. Array arguments ..
      DOUBLE PRECISION A(M,N)
C
C  =====================================================================
C
C     .. Local scalars ..
      INTEGER I,J
C     ..
C     .. Parameters ..
      DOUBLE PRECISION ZERO
      PARAMETER (ZERO=0.0D+0)
C
C     A := 0
C
      DO 10 J = 1,N
        DO 10 I = 1,M
   10     A(I,J) = ZERO
C
      RETURN
C
      END
