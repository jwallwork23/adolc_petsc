c  =====================================================================
      subroutine trans(m,n,a,c)
c
c     scalar arguments
      integer m,n
c
c     array arguments
      double precision a(m,n),c(n,m)
c
c  =====================================================================
c
c     local scalars
      double precision temp
      integer i,j
c
c     c := a**t
c
      do 10 j = 1,n
        do 10 i = 1,m
          temp = a(i,j)
   10       c(j,i) = temp
c
      return
c
      end
