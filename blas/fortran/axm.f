c  =====================================================================
c
      subroutine axm(m,n,a,lda,alpha)
c
c     .. scalar arguments ..
      double precision alpha
      integer m,n,lda
c
c     .. array arguments ..
      double precision a(lda,*)
c
c  =====================================================================
c
c     .. local scalars ..
      double precision temp
      integer i,j
c
c     a := alpha*a
c
      do 10 j = 1,n
          do 20 i = 1,m
              temp = alpha*a(i,j)
              a(i,j) = temp
   20     continue
   10 continue
c
      return
c
      end
