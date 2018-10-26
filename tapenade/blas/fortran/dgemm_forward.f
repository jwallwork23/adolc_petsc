C  =====================================================================
      SUBROUTINE DGEMM_FORWARD(TRANSA,TRANSB,M,N,K,ALPHA,A,AD,LDA,B,BD,
     +                         LDB,BETA,C,CD,LDC)
C
      INCLUDE 'dgemm.f'
      INCLUDE 'zero.f'
C
C     .. Scalar arguments ..
      DOUBLE PRECISION ALPHA,BETA
      INTEGER K,LDA,LDB,LDC,M,N
      CHARACTER TRANSA,TRANSB
C
C     .. Array arguments ..
      DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
      DOUBLE PRECISION AD(LDA,*),BD(LDB,*),CD(LDC,*)
C
C  =====================================================================
C     ..
C     .. External Subroutines ..
      EXTERNAL ZEROOUT,DGEMM
C     ..
C     .. Parameters ..
      DOUBLE PRECISION ONE
      PARAMETER (ONE=1.0d+0)
C     ..
C
C     Undifferentiated function call
C
      CALL ZEROOUT(M,N,CD,LDC)
      CALL DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
C
C     Differentiated function call
C
      CALL DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,BD,LDB,BETA,CD,LDC)
      CALL DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,AD,LDA,B,LDB,ONE,CD,LDC)
C
      RETURN
C
      END
