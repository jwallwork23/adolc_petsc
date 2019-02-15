c  =====================================================================
      subroutine dgemmr(transa,transb,m,n,k,alpha,a,ab,lda,b,bb,ldb,
     +                         beta,c,cb,ldc)
c
      include "zero.f"
      include "trans.f"
c
c     .. scalar arguments ..
      double precision alpha,beta
      integer k,lda,ldb,ldc,m,n
      character transa,transb
c
c     .. array arguments ..
      double precision a(lda,*),b(ldb,*),c(ldc,*)
      double precision ab(lda,*),bb(ldb,*),cb(ldc,*)
c
c  ====================================================================
c
c     ..
c     .. external functions ..
      logical lsame
      external lsame
c     ..
c     .. external subroutines ..
      external trans,zerout,dgemm,trans
c     ..
c     .. arrays ..
      double precision tempab(k,m),tempbb(n,k)
c     ..
c     .. logical scalars ..
      character notat,notbt
      logical nota,notb
c     ..
c     .. parameters ..
      double precision one
      parameter (one=1.0d+0)
c     ..
c
      nota = lsame(transa,'n')
      notb = lsame(transb,'n')
      if (nota) then
          notat = 'n'
      else
          notat = 't'
      end if
      if (notb) then
          notbt = 'n'
      else
          notbt = 't'
      end if

      call zerout(m,k,ab)
      call zerout(k,n,bb)
      tempbb = bb
      if (nota) then
        call dgemm('n',notbt,m,k,n,alpha,cb,m,b,ldb,one,ab,m)
      else
        call trans(m,k,ab,tempab)
        call dgemm('n',notbt,m,k,n,alpha,cb,m,b,ldb,one,tempab,k)
        call trans(k,m,tempab,ab)
      end if
      if (notb) then
        call dgemm(notat,'n',k,n,m,alpha,a,lda,cb,ldc,1,bb,n)
      else
        call trans(k,n,bb,tempbb)
        call dgemm(notat,'n',k,n,m,alpha,a,lda,cb,ldc,1,tempbb,n)
        call trans(n,k,tempbb,bb)
      end if
c
      return
c
      end
