C  =====================================================================
      SUBROUTINE DGEMMR(TRANSA,TRANSB,M,N,K,ALPHA,A,AB,LDA,B,BB,LDB,
     +                         BETA,C,CB,LDC)
C
      INCLUDE "zero.f"
      INCLUDE "trans.f"
C
C     .. Scalar arguments ..
      DOUBLE PRECISION ALPHA,BETA
      INTEGER K,LDA,LDB,LDC,M,N
      CHARACTER TRANSA,TRANSB
C
C     .. Array arguments ..
      DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
      DOUBLE PRECISION AB(LDA,*),BB(LDB,*),CB(LDC,*)
C
C  ====================================================================
C
C     ..
C     .. External functions ..
      LOGICAL LSAME
      EXTERNAL LSAME
C     ..
C     .. External subroutines ..
      EXTERNAL TRANS,ZEROUT,DGEMM,TRANS
C     ..
C     .. Arrays ..
      DOUBLE PRECISION TEMPAB(K,M),TEMPBB(N,K)
C     ..
C     .. Logical scalars ..
      CHARACTER NOTAT,NOTBT
      LOGICAL NOTA,NOTB
C     ..
C     .. Parameters ..
      DOUBLE PRECISION ONE
      PARAMETER (ONE=1.0D+0)
C     ..
C
      NOTA = LSAME(TRANSA,'N')
      NOTB = LSAME(TRANSB,'N')
      IF (NOTA) THEN
          NOTAT = 'N'
      ELSE
          NOTAT = 'T'
      END IF
      IF (NOTB) THEN
          NOTBT = 'N'
      ELSE
          NOTBT = 'T'
      END IF

      CALL ZEROUT(M,K,AB)
      CALL ZEROUT(K,N,BB)
      TEMPBB = BB
      IF (NOTA) THEN
        CALL DGEMM('N',NOTBT,M,K,N,ALPHA,CB,M,B,LDB,ONE,AB,M)
      ELSE
        CALL TRANS(M,K,AB,TEMPAB)
        CALL DGEMM('N',NOTBT,M,K,N,ALPHA,CB,M,B,LDB,ONE,TEMPAB,K)
        CALL TRANS(K,M,TEMPAB,AB)
      END IF
      IF (NOTB) THEN
        CALL DGEMM(NOTAT,'N',K,N,M,ALPHA,A,LDA,CB,LDC,1,BB,N)
      ELSE
        CALL TRANS(K,N,BB,TEMPBB)
        CALL DGEMM(NOTAT,'N',K,N,M,ALPHA,A,LDA,CB,LDC,1,TEMPBB,N)
        CALL TRANS(N,K,TEMPBB,BB)
      END IF
C
      RETURN
C
      END
