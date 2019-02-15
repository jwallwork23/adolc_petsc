c =====================================================================
c
      include "trans.f"
      include "zero.f"
c
c =====================================================================
c
c     external subroutines
      external trans,zerout,dgemm
c     scalars
      integer m,n
      parameter (m=3,n=5)
      parameter (one=1.0d+0,zero=0.0d+0)
      character transa,transb
c
c     arrays
      double precision a(m,n),b(n,m),c(m,m)
c
c     supply values in storage order
      data a/
     1     1, 2, 3,
     2     4, 5, 6,
     3     7, 8, 9,
     4     10, 11,12,
     5     13,14,15/
c
      write (6,*) "a:"
      write (6,1) ((a(i,j),j = 1,n),i = 1,m)
c
c     test 1: transpose function
c
      write (6,*)
      write (6,*) "b = transpose(a):"
c
      call trans(m,n,a,b)
c
      write (6,2) ((b(i,j),j = 1,m),i = 1,n)
c
c     test 2: dgemm  fixme
c
      transa = "n"
      transb = "n"
      call dgemm(transa,transb,m,m,n,one,a,m,b,n,zero,c,m)
c
      write (6,*)
      write (6,*) "c = a*b:"
      write (6,2) ((c(i,j),j = 1,m),i = 1,m)
c
c     test 3: zeroing function
c
      call zerout(m,n,a)
c
      write (6,*)
      write (6,*) "a = 0:"
      write (6,1) ((a(i,j),j = 1,n),i = 1,m)
c
c     five and three values per line, resp.
    1 format (5f6.1)
    2 format (3f6.1)
      end
