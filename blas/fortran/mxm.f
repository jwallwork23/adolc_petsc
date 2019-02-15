c  =====================================================================
c
      subroutine mxm(m,n,k,a,lda,b,ldb,c,ldc)
c
c     scalar arguments
      integer k,m,n,lda,ldb,ldc
c
c     array arguments
      double precision a(lda,*),b(ldb,*),c(ldc,*)
c
c  =====================================================================
c
c     local scalars
      double precision temp
      integer i,j,l
c
c     c := a * b
c
      do 10 j = 1,n
          do 20 l = 1,k
              temp = b(l,j)
              do 30 i = 1,m
                  c(i,j) = c(i,j) + temp*a(i,l)
   30         continue
   20     continue
   10 continue
c
      return
c
      end
