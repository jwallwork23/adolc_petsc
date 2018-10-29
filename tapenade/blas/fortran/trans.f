C  =====================================================================
      SUBROUTINE TRANS(M,N,A,C)
C
C     Scalar arguments
      INTEGER M,N
C
C     Array arguments
      DOUBLE PRECISION A(M,N),C(N,M)
C
C  =====================================================================
C
C     Local scalars
      DOUBLE PRECISION TEMP
      INTEGER I,J
C
C     C := A**T
C
      DO 10 J = 1,N
        DO 10 I = 1,M
          TEMP = A(I,J)
   10       C(J,I) = TEMP
C
      RETURN
C
      END
