!----------------------------------------------------------------------
!----------------------------------------------------------------------
!   BR :        The Beeler - Reuter model
!----------------------------------------------------------------------
!----------------------------------------------------------------------

PURE FUNCTION ab(c1,c2,c3,c4,c5,c6,c7,V)
    DOUBLE PRECISION, INTENT(IN) :: c1,c2,c3,c4,c5,c6,c7,V
    DOUBLE PRECISION :: ab
    ab = (c1*exp(c2*(V+c3))+c4*(V+c5))/(exp(c6*(V+c3))+c7)
END FUNCTION

PURE FUNCTION cas(Ca)
    DOUBLE PRECISION, INTENT(IN) :: Ca
    DOUBLE PRECISION :: cas, ep
    ep = 1.5d-10
    cas = ep*0.5d0*(1.0d0 + TANH(1.0d0-(Ca/ep))) + Ca*0.5d0*(1.0d0 + TANH((Ca/ep)-1.0))
END FUNCTION

PURE FUNCTION Istim(Y,PAR)
	DOUBLE PRECISION, INTENT(IN) :: Y, PAR(*)
	DOUBLE PRECISION :: Istim
	Istim = PAR(1)*(Y**PAR(3))
END FUNCTION

SUBROUTINE FUNC(NDIM,X,ICP,PAR,IJAC,F,DFDU,DFDP)

	! the main dynamics equation

	IMPLICIT NONE
	INTEGER, INTENT(IN) :: NDIM, ICP(*), IJAC
	DOUBLE PRECISION, INTENT(IN) :: X(NDIM), PAR(*)
	DOUBLE PRECISION, INTENT(OUT) :: F(NDIM)
	DOUBLE PRECISION, INTENT(INOUT) :: DFDU(NDIM,NDIM), DFDP(NDIM,*)

	DOUBLE PRECISION TPI
	DOUBLE PRECISION A,B,p

	DOUBLE PRECISION ab,cas

	DOUBLE PRECISION ax,bx,am,bm,ah,bh,aj,bj,ad,bd,af,bf,IK,Ix,INa,Is

	TPI=8*ATAN(1.0D0)

	A=PAR(1)
	B=PAR(2)
	p=PAR(3)
    
	ax=ab( 0.0005d0, 0.083d0, 50.0d0, 0.0d0, 0.0d0, 0.057d0, 1.0d0, X(1))
	bx=ab( 0.0013d0,-0.06d0 , 20.0d0, 0.0d0, 0.0d0,-0.04d0 , 1.0d0, X(1))
	am=ab( 0.0d0   , 0.0d0  , 47.0d0,-1.0d0,47.0d0,-0.1d0  ,-1.0d0, X(1))
	bm=ab(40.0d0   ,-0.056d0, 72.0d0, 0.0d0, 0.0d0, 0.0d0  , 0.0d0, X(1))
	ah=ab( 0.126d0 ,-0.25d0 , 77.0d0, 0.0d0, 0.0d0, 0.0d0  , 0.0d0, X(1))
	bh=ab( 1.7d0   , 0.0d0  , 22.5d0, 0.0d0, 0.0d0,-0.082d0, 1.0d0, X(1))
	aj=ab( 0.055d0 ,-0.25d0 , 78.0d0, 0.0d0, 0.0d0,-0.2d0  , 1.0d0, X(1))
	bj=ab( 0.3d0   , 0.0d0  , 32.0d0, 0.0d0, 0.0d0,-0.1d0  , 1.0d0, X(1))
	ad=ab( 0.095d0 ,-0.01d0 , -5.0d0, 0.0d0, 0.0d0,-0.072d0, 1.0d0, X(1))
	bd=ab( 0.07d0  ,-0.017d0, 44.0d0, 0.0d0, 0.0d0, 0.05d0 , 1.0d0, X(1))
	af=ab( 0.012d0 ,-0.008d0, 28.0d0, 0.0d0, 0.0d0, 0.15d0 , 1.0d0, X(1))
	bf=ab( 0.0065d0,-0.02d0 , 30.0d0, 0.0d0, 0.0d0,-0.2d0  , 1.0d0, X(1))
    
	IK=(exp(0.08d0*(X(1)+53.0d0)) + exp(0.04d0*(X(1)+53.0d0)))
	IK=4.0d0*(exp(0.04d0*(X(1)+85.0d0)) - 1.0d0)/IK
	IK=IK+0.2d0*(X(1)+23.0d0)/(1.0d0-exp(-0.04d0*(X(1)+23.0d0)))
	IK=0.35d0*IK
	Ix=exp(0.04d0*(X(1)+35.0d0))
	Ix=X(3)*0.8d0*(exp(0.04d0*(X(1)+77.0d0))-1.0d0)/Ix
	INa=(4.0d0*X(4)*X(4)*X(4)*X(5)*X(6) + 0.003d0)*(X(1)-50.0d0)
	Is=0.09d0*X(7)*X(8)*(X(1)+82.3d0+13.0287d0*log(cas(X(2))))
    
	F(1)= PAR(1)*(X(9)**PAR(3)) - (IK + Ix + INa + Is)
	F(2)= -1.0d-7 * Is + 0.07d0*(1.0d-7 - X(2))
	F(3)= ax*(1.0d0 - X(3)) - bx*X(3)
	F(4)= am*(1.0d0 - X(4)) - bm*X(4)
	F(5)= ah*(1.0d0 - X(5)) - bh*X(5)
	F(6)= aj*(1.0d0 - X(6)) - bj*X(6)
	F(7)= ad*(1.0d0 - X(7)) - bd*X(7)
	F(8)= af*(1.0d0 - X(8)) - bf*X(8)

	F(9)= X(9) + (TPI*B/1000.0)*X(10) - X(9) *(X(9)*X(9) + X(10)*X(10))
	F(10)=X(10)- (TPI*B/1000.0)*X(9)  - X(10)*(X(9)*X(9) + X(10)*X(10))

	IF(IJAC.EQ.0)RETURN

	! DFDU definition goes here

	IF(IJAC.EQ.1)RETURN

	! DFDP definition goes here

END SUBROUTINE FUNC
!----------------------------------------------------------------------
!----------------------------------------------------------------------

SUBROUTINE STPNT(NDIM,U,PAR,T)
	!     ---------- -----

	IMPLICIT NONE
	INTEGER, INTENT(IN) :: NDIM
	DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
	DOUBLE PRECISION, INTENT(IN) :: T

	DOUBLE PRECISION TPI

	TPI=8*ATAN(1.0D0)

	!
	! Initialize the equation parameters
	PAR(1)=0.0d0
	PAR(2)=TPI/1000.0d0
	PAR(3)=1.0d0

	! Initialize the solution (assuming PAR(1)=0 )
	! rest state for U(1:8)
	! exact periodic orbit for U(9:10)
	U(1)=-84.57375612224915d0
	U(2)=1.7820072156205916d-7
	U(3)=0.00562865057175647d0
	U(4)=0.010981968723276314d0
	U(5)=0.9877211754875368d0
	U(6)=0.9748381389815896d0
	U(7)=0.002970724663210886d0
	U(8)=0.9999813338936878d0

	U(9)=SIN(TPI*T)
	U(10)=COS(TPI*T)

END SUBROUTINE STPNT
!----------------------------------------------------------------------
!----------------------------------------------------------------------
SUBROUTINE BCND
END SUBROUTINE BCND

SUBROUTINE ICND
END SUBROUTINE ICND

SUBROUTINE FOPT
END SUBROUTINE FOPT

SUBROUTINE PVLS
END SUBROUTINE PVLS
!----------------------------------------------------------------------
!----------------------------------------------------------------------
