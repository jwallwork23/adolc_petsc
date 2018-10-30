C  =====================================================================
C
      SUBROUTINE AXM(M,N,A,LDA,ALPHA)
C
C     .. Scalar arguments ..
      DOUBLE PRECISION ALPHA
      INTEGER M,N,LDA
C
C     .. Array arguments ..
      DOUBLE PRECISION A(LDA,*)
C
C  =====================================================================
C
C     .. Local scalars ..
      DOUBLE PRECISION TEMP
      INTEGER I,J
C
C     A := alpha*A
C
      DO 10 J = 1,N
          DO 20 I = 1,M
              TEMP = ALPHA*A(I,J)
              A(I,J) = TEMP
   20     CONTINUE
   10 CONTINUE
C
      RETURN
C
      END
