"""Out-of-sample replication on governor races (never used in fitting). Parameters are frozen at Senate values.
Tests: (A) trailing incumbents noisier? (B) leads growing beyond polls (C) independents running behind polls."""
import sys;sys.path.insert(0,'models')
import numpy as np,pandas as pd
from scipy import stats
from core import aggregate
P=pd.read_csv('data/raw/fte/raw_polls.csv');G=P[P.type_simple=='Gov-G'].copy()
adj=(2.12,-2.15)
G['age']=G.time_to_election;G['N']=pd.to_numeric(G.samplesize,errors='coerce');G['part']=G.partisan.fillna('')
G['und']=100-G.cand1_pct-G.cand2_pct
def D(r):
    if r.cand1_party=='DEM' and r.cand2_party=='REP':return r.cand1_pct-r.cand2_pct,r.cand1_actual-r.cand2_actual,r.cand1_name,r.cand2_name
    if r.cand2_party=='DEM' and r.cand1_party=='REP':return r.cand2_pct-r.cand1_pct,r.cand2_actual-r.cand1_actual,r.cand2_name,r.cand1_name
    return np.nan,np.nan,None,None
x=G.apply(D,axis=1,result_type='expand');G['m'],G['act'],G['dn'],G['rn']=x[0],x[1],x[2],x[3]
Gd=G[G.m.notna()]
rows=[]
for (y,st,race),g in Gd.groupby(['cycle','location','race']):
    a=aggregate(g[['age','N','part','m','und','pollster']],adj)
    rows.append(dict(year=y,st=st,act=g.act.iloc[0],dn=g.dn.iloc[0],rn=g.rn.iloc[0],**a))
R=pd.DataFrame(rows).dropna(subset=['p'])
# incumbency: winner of a previous governor race in that state within 4 yrs (misses successors/appointees -> treated as open)
win={}
for r in R.sort_values('year').itertuples():
    win.setdefault(r.st,[]).append((r.year,r.dn if r.act>0 else r.rn))
def inc(r):
    pri=[n for y,n in win.get(r.st,[]) if 0<r.year-y<=4]
    return 1 if r.dn in pri else (-1 if r.rn in pri else 0)
R['inc']=R.apply(inc,axis=1)
g=lambda m:m*(1+.13*np.exp(-(m/20.)**2));R['mg']=g(R.p)
R['T']=np.where((R.inc==1)&(R.mg<0)&(R.mg>-10),1,np.where((R.inc==-1)&(R.mg>0)&(R.mg<10),-1,0))
R['lw']=(np.sign(R.act)==np.sign(R.mg)).astype(int);R['y']=(R.act>0)*1.
print('governor races',len(R),'cycles',R.year.nunique(),'first',R.year.min(),'last',R.year.max(),'incumbent-flagged',(R.inc!=0).sum(),'trailing',(R['T']!=0).sum())
rng=np.random.default_rng(1)
def cb(v,yrs,B=4000):
    u=np.unique(yrs);gp={a:v[yrs==a] for a in u};o=[np.concatenate([gp[a] for a in rng.choice(u,len(u))]).mean() for _ in range(B)];return np.quantile(o,[.05,.95])
def rep(nm,d):
    if not len(d):
        print(nm,'n=0');return
    ci=cb(d.lw.values,d.year.values);print(f"{nm:44s} n={len(d):3d} cyc={d.year.nunique():2d} leader won {d.lw.mean():.3f} [{ci[0]:.2f},{ci[1]:.2f}]")
c=R[(R.mg.abs()>=2)&(R.mg.abs()<8)]
rep('All leads 2-8',c);rep('Leads 2-5',R[(R.mg.abs()>=2)&(R.mg.abs()<5)]);rep('Leads 5-8',R[(R.mg.abs()>=5)&(R.mg.abs()<8)])
rep('Challenger leads incumbent 2-8',c[c['T']!=0]);rep('  ...2-5',c[(c['T']!=0)&(c.mg.abs()<5)]);rep('  ...5-8',c[(c['T']!=0)&(c.mg.abs()>=5)])
rep('Open seat or no incumbent 2-8',c[c['T']==0]);rep('  ...2-5',c[(c['T']==0)&(c.mg.abs()<5)]);rep('  ...5-8',c[(c['T']==0)&(c.mg.abs()>=5)])
rep('Incumbent leads 2-8',c[(c.inc!=0)&(np.sign(c.inc)==np.sign(c.mg))])
# mean error toward leader
for nm,d in (('trailing-inc',c[c['T']!=0]),('other',c[c['T']==0])):print(nm,'mean(act-mg) toward leader %.2f'%np.mean((d.act-d.mg)*np.sign(d.mg)))
# (A) frozen-parameter comparison
def logit(p):p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
def expit(x):return 1/(1+np.exp(-x))
def ll(p,y):p=np.clip(p,.003,.997);return -(y*np.log(p)+(1-y)*np.log(1-p))
sd=np.sqrt(3.15**2+3.62**2/np.maximum(R.neff,.2)+(.15*R.p)**2+2.43**2+(.6*0)**2);x=logit(stats.norm.cdf(R.mg/sd));TT=(R['T']!=0).values
base=expit(1.345*x);noisy=expit(np.where(TT,.674,1.686)*x)
lb,ln=ll(base,R.y.values),ll(noisy,R.y.values)
print('FROZEN SENATE PARAMS on governors: base LL %.4f, noisier LL %.4f | Brier %.4f %.4f'%(lb.mean(),ln.mean(),np.mean((base-R.y)**2),np.mean((noisy-R.y)**2)))
print('  in trailing races n=%d: base LL %.3f noisier LL %.3f; leader rate %.3f, base says %.3f, noisier says %.3f'%(TT.sum(),lb[TT].mean(),ln[TT].mean(),R.lw[TT].mean(),np.maximum(base[TT],1-base[TT]).mean(),np.maximum(noisy[TT],1-noisy[TT]).mean()))
dd=pd.Series(ln-lb).groupby(R.year.values).sum();bs=[dd.sample(len(dd),replace=True,random_state=int(rng.integers(1e9))).sum() for _ in range(5000)];print('  noisier-base total %.2f 90%%CI [%.2f,%.2f]'%(dd.sum(),*np.quantile(bs,[.05,.95])))
# (B) expansion
import statsmodels.api as sm
for lim in (10,15):
    d=R[R.mg.abs()<lim];m=sm.OLS(d.act,sm.add_constant(d[['p']])).fit(cov_type='cluster',cov_kwds={'groups':d.year});print(f'act~p |mg|<{lim} n={len(d)} beta {m.params["p"]:.3f} se {m.bse["p"]:.3f}')
# (C) independents vs major party in governor + senate, top-two head to head
I=P[P.type_simple.isin(['Gov-G','Sen-G'])].copy()
ism=(I.cand1_party=='IND')^(I.cand2_party=='IND');I=I[ism&I.cand1_party.isin(['IND','REP','DEM'])&I.cand2_party.isin(['IND','REP','DEM'])]
I['im']=np.where(I.cand1_party=='IND',I.cand1_pct-I.cand2_pct,I.cand2_pct-I.cand1_pct);I['ia']=np.where(I.cand1_party=='IND',I.cand1_actual-I.cand2_actual,I.cand2_actual-I.cand1_actual);I['iname']=np.where(I.cand1_party=='IND',I.cand1_name,I.cand2_name);I['opp']=np.where(I.cand1_party=='IND',I.cand2_party,I.cand1_party)
T=I[I.time_to_election<=30].groupby(['race','cycle','location','iname','opp']).agg(polls=('im','size'),poll=('im','mean'),act=('ia','first')).reset_index();T['err']=T.act-T.poll
T=T[T.poll.abs()<25];print(T.sort_values('cycle')[['race','iname','opp','polls','poll','act','err']].round(1).to_string())
print('independents |poll|<25: n=%d mean err %.2f median %.2f se %.2f; with n>=3 polls: n=%d mean %.2f se %.2f'%(len(T),T.err.mean(),T.err.median(),T.err.std()/np.sqrt(len(T)),(T.polls>=3).sum(),T[T.polls>=3].err.mean(),T[T.polls>=3].err.std()/np.sqrt((T.polls>=3).sum())))
T.to_csv('output/independent_candidates_all.csv',index=False);R.to_csv('data/processed/governor_replication.csv',index=False)
