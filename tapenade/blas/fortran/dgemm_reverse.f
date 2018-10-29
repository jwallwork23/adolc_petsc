C  =====================================================================
      SUBROUTINE DGEMMR(TRANSA,TRANSB,M,N,K,ALPHA,A,AB,LDA,B,BB,LDB,
     +                         BETA,C,CB,LDC)
C
      INCLUDE 'dgemm.f'
      INCLUDE 'zero.f'
      INCLUDE 'trans.f'
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
      EXTERNAL ZEROUT,DGEMM,TRANS
C     ..
C     .. Arrays ..
      DOUBLE PRECISION TEMPAB(LDA,*),TEMPBB(LDB,*)
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

      CALL ZEROUT(M,P,AB,LDA)
      CALL ZEROUT(P,N,BB,LDB)
      TEMPAB = AB
      TEMPBB = BB
      CALL DGEMM('N',NOTBT,M,K,N,ALPHA,CB,LDC,B,LDB,ONE,TEMPAB,LDA)
      CALL DGEMM(NOTAT,'N',P,N,M,ALPHA,A,LDA,CB,LDC,1,TEMPBB,LDB)
      IF .NOT. (NOTA)
          CALL TRANS(M,P,TEMPAB,AB)
      END IF
      IF .NOT. (NOTB)
          CALL TRANS(P,N,TEMPBB,BB)
      END IF
C
      RETURN
C
      END
