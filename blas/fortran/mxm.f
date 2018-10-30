C  =====================================================================
C
      SUBROUTINE MXM(M,N,K,A,LDA,B,LDB,C,LDC)
C
C     Scalar arguments
      INTEGER K,M,N,LDA,LDB,LDC
C
C     Array arguments
      DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
C
C  =====================================================================
C
C     Local scalars
      DOUBLE PRECISION TEMP
      INTEGER I,J,L
C
C     C := A * B
C
      DO 10 J = 1,N
          DO 20 L = 1,K
              TEMP = B(L,J)
              DO 30 I = 1,M
                  C(I,J) = C(I,J) + TEMP*A(I,L)
   30         CONTINUE
   20     CONTINUE
   10 CONTINUE
C
      RETURN
C
      END
