"""Memory-one (opponent-reactive) strategies in the iterated Prisoner's Dilemma with payoffs T,R,P,S = 5,3,1,0.
A strategy is (alpha, gamma): gamma = P(C | opponent played C), alpha = P(C | opponent played D).
Stationary cooperation rates (Nowak & Sigmund): c1 = (a1 + r1 a2)/(1 - r1 r2), c2 = (a2 + r2 a1)/(1 - r1 r2), r = gamma - alpha;
the two players' moves are independent in the stationary state because the chain splits into two interleaved chains.
(1) Symbolically: at a symmetric profile dU1/dalpha1 = -(1-g) Q / D and dU1/dgamma1 = -a Q / D, with D = (1-r)^3 (1+r) and
    Q = 3a^2 - 7ag + 6a + 4g^2 - 5g + 1. Against (a, g) player 1's payoff depends on her strategy only through her rate c, as
    W(c) = 1 + 4a + (4r-1-a) c - r c^2, and (1-r) W'(c0) = -Q at c0 = a/(1-r), so Q(0,1) = 0 is the cleared denominator of r = 1.
    The arc of Q = 0 inside the unit square is c = (4r-1)/(1+r), a = c(1-r), g = a + r for r in [1/4, 2/3].
    For general payoffs the arc meets gamma = 1 at a* = (2R-T-S)/(R-S); the same computation with 10,5,1,0 gives equation 11
    of Menshikov et al. 2017, whose arc ends at (0,1) because 2R = T+S.
(2) Numerically: on the claimed equilibrium set (alpha = 0, gamma <= 1/4; the arc; gamma = 1, alpha <= 1/3; tit-for-tat opening
    with C) no own strategy on a 401x401 grid beats the symmetric value; on a 401x401 grid of symmetric profiles every profile
    without a profitable deviation lies next to that set.
Writes numbers_curve.tex."""
import sympy as sp, numpy as np
a1,g1,a,g=sp.symbols('alpha1 gamma1 alpha gamma', real=True); T,R,P,S=5,3,1,0
r1=g1-a1; r=g-a; den=1-r1*r; c1=(a1+r1*a)/den; c2=(a+r*a1)/den
U=R*c1*c2+S*c1*(1-c2)+T*(1-c1)*c2+P*(1-c1)*(1-c2)
Q=3*a**2-7*a*g+6*a+4*g**2-5*g+1; D=(1-r)**3*(1+r)
assert sp.simplify(sp.diff(U,a1).subs({a1:a,g1:g})+(1-g)*Q/D)==0 and sp.simplify(sp.diff(U,g1).subs({a1:a,g1:g})+a*Q/D)==0
c,t=sp.symbols('c r')
def W_of(T,R,P,S,al,ga):
    o=al+(ga-al)*c; return R*c*o+S*c*(1-o)+T*(1-c)*o+P*(1-c)*(1-o)
W=W_of(T,R,P,S,a,g)
assert sp.expand(W-(1+4*a+(4*r-1-a)*c-r*c**2))==0
assert sp.simplify((1-r)*sp.diff(W,c).subs(c,a/(1-r))+Q)==0
ca=(4*t-1)/(1+t); aa=ca*(1-t); assert sp.simplify(Q.subs({a:aa,g:aa+t},simultaneous=True))==0
assert sp.simplify(ca.subs(t,sp.Rational(1,4)))==0 and sp.simplify(ca.subs(t,sp.Rational(2,3)))==1
Ts,Rs,Ps,Ss=sp.symbols('T R P S'); astar=sp.solve(sp.diff(W_of(Ts,Rs,Ps,Ss,a,1),c).subs(c,1),a)
assert len(astar)==1 and sp.simplify(astar[0]-(2*Rs-Ts-Ss)/(Rs-Ss))==0
WH=W_of(10,5,1,0,a,g); QH=5*a**2+9*g**2-14*a*g+14*a-10*g+1
assert sp.simplify((1-r)*sp.diff(WH,c).subs(c,a/(1-r))+QH)==0
print("derivative formula, W, the arc, a* and equation 11 of Menshikov et al. 2017: symbolic checks passed")

Uf=sp.lambdify((a1,g1,a,g),U,'numpy')
grid=np.linspace(0,1,401); A1,G1=np.meshgrid(grid,grid,indexing='ij')
def own_best(av,gv):
    with np.errstate(divide='ignore',invalid='ignore'): V=Uf(A1,G1,av,gv)
    return float(V[np.isfinite(V)].max())
rs=np.linspace(0.25,2/3,41); ca_=(4*rs-1)/(1+rs); arc=np.c_[ca_*(1-rs),ca_*(1-rs)+rs]
low=np.c_[np.zeros(11),np.linspace(0,0.25,11)]; top=np.c_[np.linspace(1/60,1/3,20),np.ones(20)]
gain=0.0
for av,gv in np.r_[arc,low,top]: gain=max(gain,own_best(av,gv)-float(Uf(av,gv,av,gv)))
tft=own_best(0.0,1.0)-R; gain=max(gain,tft)        # tit-for-tat opening with C earns R
print(f"on {len(arc)} arc points and {len(low)+len(top)} segment points plus tit-for-tat, the largest gain from any own strategy on a 401x401 grid is {gain:.2e}")
print(f"tit-for-tat: gain {tft:.2e} after opening CC, {own_best(0.0,1.0)-P:.2f} after DD, {own_best(0.0,1.0)-2.25:.2f} after CD/DC")
for av,gv in [(1.0,1.0),(0.34,1.0),(0.0,0.26)]: print(f"just outside: ({av},{gv}) gain {own_best(av,gv)-float(Uf(av,gv,av,gv)):.4f}")
rf=np.linspace(0.25,2/3,200001); cf=(4*rf-1)/(1+rf); af=cf*(1-rf); i=int(af.argmax())
print(f"arc: alpha peaks at {af[i]:.4f}, gamma {af[i]+rf[i]:.4f}")

# Every symmetric grid profile: gain against itself, using the exact maximum of W over c in [0,1].
Sa,Sg=np.meshgrid(grid,grid,indexing='ij'); Rr=Sg-Sa
def Wn(cv,av,rv): return 1+4*av+(4*rv-1-av)*cv-rv*cv*cv
with np.errstate(divide='ignore',invalid='ignore'):
    cin=np.where(Rr>0,np.clip((4*Rr-1-Sa)/(2*np.where(Rr>0,Rr,1)),0,1),0.0); c0=np.where(Rr<1,Sa/(1-np.where(Rr<1,Rr,0)),1.0)
best=np.maximum.reduce([Wn(cin,Sa,Rr),Wn(0.0,Sa,Rr),Wn(1.0,Sa,Rr)]); G=best-Wn(c0,Sa,Rr)
rng=np.random.default_rng(0); idx=rng.integers(0,401,(200,2))
for i_,j_ in idx:                                  # the reduction to W agrees with the search over own strategies
    if Rr[i_,j_]<1: assert abs(own_best(Sa[i_,j_],Sg[i_,j_])-best[i_,j_])<2e-3,(Sa[i_,j_],Sg[i_,j_])
tol=1e-4; eq=np.c_[Sa[G<tol],Sg[G<tol]]
ref=np.r_[np.c_[af[::20],af[::20]+rf[::20]],np.c_[np.zeros(2001),np.linspace(0,0.25,2001)],np.c_[np.linspace(0,1/3,2001),np.ones(2001)]]
dist=max(float(np.hypot(*(ref-e).T).min()) for e in eq)
print(f"symmetric grid: {len(eq)} of {G.size} profiles have gain below {tol:g}; the farthest of them lies {dist:.4f} from the claimed set")
open(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)),'numbers_curve.tex'),'w').write("\\newcommand{\\nCurvePoints}{%d}\\newcommand{\\nSegmentPoints}{%d}\\newcommand{\\curveGain}{%.1e}\\newcommand{\\gridDist}{%.3f}\n"%(len(arc),len(low)+len(top),max(gain,0.0),dist))
