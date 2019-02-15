c  =====================================================================
      subroutine dgemmf(transa,transb,m,n,k,alpha,a,ad,lda,b,bd,ldb,
     +                         beta,c,cd,ldc)
c
      include "zero.f"
c
c     .. scalar arguments ..
      double precision alpha,beta
      integer k,lda,ldb,ldc,m,n
      character transa,transb
c
c     .. array arguments ..
      double precision a(lda,*),b(ldb,*),c(ldc,*)
      double precision ad(lda,*),bd(ldb,*),cd(ldc,*)
c
c  =====================================================================
c     ..
c     .. external subroutines ..
      external zerout,dgemm
c     ..
c     .. parameters ..
      double precision one
      parameter (one=1.0d+0)
c     ..
c
c     undifferentiated function call
c
      call zerout(m,n,cd)
      call dgemm(transa,transb,m,n,k,alpha,a,lda,b,ldb,beta,c,ldc)
c
c     differentiated function call
c
      call dgemm(transa,transb,m,n,k,alpha,a,lda,bd,ldb,beta,cd,ldc)
      call dgemm(transa,transb,m,n,k,alpha,ad,lda,b,ldb,one,cd,ldc)
c
      return
c
      end
