import numpy as np,pandas as pd
from scipy import stats,optimize
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def expit(x): return 1/(1+np.exp(-x))
def br(p,y): return np.mean((p-y)**2)
def ll(p,y): p=np.clip(p,.003,.997); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
B=pd.read_csv('data/processed/backtest_final.csv'); y=(B.margin>0).values.astype(float); base=stats.norm.cdf(B.mg/B.sd)
TT=(B['T']!=0).values; x=logit(base)
def nl(pr,y): pr=np.clip(pr,1e-6,1-1e-6); return -np.sum(y*np.log(pr)+(1-y)*np.log(1-pr))
def P_k(par,i): return expit(par[0]*x[i])
def P_ktT(par,i): return expit(np.where(TT[i],par[1],par[0])*x[i])
def P_delta(par,i): return expit(par[0]*logit(stats.norm.cdf((B.mg.values[i]+par[1]*B['T'].values[i])/B.sd.values[i])))
vs={'base k':(P_k,[1.4],[(.5,4)]),'k + k_T (noisier trailing-inc)':(P_ktT,[1.4,1.],[(.5,4),(.3,4)]),'k + delta (shifted)':(P_delta,[1.4,1.],[(.5,4),(-3,10)])}
res={}
for nm,(fn,x0,bd) in vs.items():
    pr=np.zeros(len(B))
    for yr in B.year.unique():
        te=(B.year==yr).values; tr=np.where(~te)[0]
        p=optimize.minimize(lambda a:nl(fn(a,tr),y[tr]),x0,method='L-BFGS-B',bounds=bd).x; pr[te]=fn(p,np.where(te)[0])
    full=optimize.minimize(lambda a:nl(fn(a,np.arange(len(B))),y),x0,method='L-BFGS-B',bounds=bd).x
    res[nm]=pr; print(f"{nm:34s} LOCO LL {ll(pr,y):.4f} Brier {br(pr,y):.4f} | in T races LL {ll(pr[TT],y[TT]):.3f} Brier {br(pr[TT],y[TT]):.3f} | full-fit params {np.round(full,3)}")
# paired cycle bootstrap for LL difference
rng=np.random.default_rng(3); a=res['base k'];b=res['k + k_T (noisier trailing-inc)']
def pll(p,y): p=np.clip(p,.003,.997); return -(y*np.log(p)+(1-y)*np.log(1-p))
dl=pd.Series(pll(b,y)-pll(a,y)).groupby(B.year.values).sum(); bs=[dl.sample(len(dl),replace=True,random_state=int(rng.integers(1e9))).sum() for _ in range(5000)]
print('k_T minus base: total LL diff %.2f, 90%%CI [%.2f, %.2f] (negative = k_T better)'%(dl.sum(),*np.quantile(bs,[.05,.95])))
# reliability inside T races per variant
for nm,pr in res.items():
    s=B[TT]; p=pr[TT]; fav=np.maximum(p,1-p); print(nm,'T-race leader-rate %.3f vs mean fav-prob %.3f'%(np.mean(np.sign(s.margin)==np.sign(s.mg)),fav.mean()))
# State poll-error history
for st in ['ME','MI','TX','OH','IA','KS','AK','SC','MS','LA','FL','NE','GA','NC','NH','MN','MT']:
    g_=B[(B.st==st)&B.p.notna()].sort_values('year'); e=(g_.margin-g_.p)
    print(st,'mean %+.1f rmse %.1f |'%(e.mean(),np.sqrt((e**2).mean())),' '.join(f"{str(a)[2:]}:{b:+.0f}" for a,b in zip(g_.year,e)))
# dossier: poll trends
P=pd.read_csv('data/processed/polls_clean.csv')
for st in ['SC','NE','ME','MI','TX','OH','KS','AK','IA','FL','MN','NH','GA','NC','MT']:
    g_=P[(P.st==st)&(P.age<=70)].sort_values('age')
    a,b,c=g_[g_.age<=10],g_[(g_.age>10)&(g_.age<=30)],g_[g_.age>30]
    print(f"{st}: <=10d n={len(a)} {a.margin.mean():+.1f} | 11-30d n={len(b)} {b.margin.mean():+.1f} | 31-70d n={len(c)} {c.margin.mean():+.1f} || last:",'; '.join(f"{x.hkey[:14]}{'('+x.partisan+')' if x.partisan==x.partisan and x.partisan else ''} {x.margin:+.0f}" for x in g_.head(7).itertuples()))
