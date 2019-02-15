c  =====================================================================
c
      subroutine zerout(m,n,a)
c
c     .. scalar arguments ..
      integer m,n
c
c     .. array arguments ..
      double precision a(m,n)
c
c  =====================================================================
c
c     .. local scalars ..
      integer i,j
c     ..
c     .. parameters ..
      double precision zero
      parameter (zero=0.0d+0)
c
c     a := 0
c
      do 10 j = 1,n
        do 10 i = 1,m
   10     a(i,j) = zero
c
      return
c
      end
