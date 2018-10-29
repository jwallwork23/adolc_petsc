C =====================================================================
C
      INCLUDE 'trans.f'
C
C =====================================================================
C      EXTERNAL TRANS
C
      INTEGER K,L
      PARAMETER (K=3,L=5)
      DOUBLE PRECISION A(K,L),B(L,K)
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
      WRITE (6,1) ((A(I,J),J = 1,L),I = 1,K)
C
      WRITE (6,*)
      WRITE (6,*) "B = Transpose(A):"
C
      CALL TRANS(K,L,A,B)
C
      WRITE (6,2) ((B(I,J),J = 1,K),I = 1,L)
C
C     Five and three values per line, resp.
    1 FORMAT (5F6.1)
    2 FORMAT (3F6.1)
      END
