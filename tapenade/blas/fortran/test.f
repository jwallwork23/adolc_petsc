C =====================================================================
C
      INCLUDE 'trans.f'
C
C =====================================================================
C      EXTERNAL TRANS
C
      DIMENSION A(3,5),B(5,3)
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
      WRITE (6,1) ((A(I,J),J = 1,5),I = 1,3)
C
      WRITE (6,*)
      WRITE (6,*) "B = Transpose(A):"
      DO 10 I = 1,3
        DO 10 J = 1,5
   10     B(J,I) = A(I,J)
C
      WRITE (6,*) "... written as B(row,column):"
      WRITE (6,2) ((B(I,J),J = 1,3),I = 1,5)
C
C     Five and three values per line, resp.
    1 FORMAT (5F6.1)
    2 FORMAT (3F6.1)
      END
