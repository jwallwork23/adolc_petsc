C =====================================================================
C
      INCLUDE "trans.f"
      INCLUDE "zero.f"
C
C =====================================================================
C
C     External subroutines
      EXTERNAL TRANS,ZEROUT,DGEMM
C     Scalars
      INTEGER M,N
      PARAMETER (M=3,N=5)
      PARAMETER (ONE=1.0D+0,ZERO=0.0D+0)
      CHARACTER TRANSA,TRANSB
C
C     Arrays
      DOUBLE PRECISION A(M,N),B(N,M),C(M,M)
C
C     Supply values in storage order
      DATA A/
     1     1, 2, 3,
     2     4, 5, 6,
     3     7, 8, 9,
     4     10, 11,12,
     5     13,14,15/
C
      WRITE (6,*) "A:"
      WRITE (6,1) ((A(I,J),J = 1,N),I = 1,M)
C
C     TEST 1: transpose function
C
      WRITE (6,*)
      WRITE (6,*) "B = Transpose(A):"
C
      CALL TRANS(M,N,A,B)
C
      WRITE (6,2) ((B(I,J),J = 1,M),I = 1,N)
C
C     TEST 2: dgemm
C
      TRANSA = "N"
      TRANSB = "N"
      CALL DGEMM(TRANSA,TRANSB,M,M,N,ONE,A,M,B,N,ZERO,C,M)
C
      WRITE (6,*)
      WRITE (6,*) "C = A*B:"
      WRITE (6,2) ((C(I,J),J = 1,M),I = 1,M)
C
C     TEST 3: zeroing function
C
      CALL ZEROUT(M,N,A,M)
C
      WRITE (6,*)
      WRITE (6,*) "A = 0:"
      WRITE (6,1) ((A(I,J),J = 1,N),I = 1,M)
C
C     Five and three values per line, resp.
    1 FORMAT (5F6.1)
    2 FORMAT (3F6.1)
      END
