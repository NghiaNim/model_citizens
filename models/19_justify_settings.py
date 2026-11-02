"""Out-of-sample tests of the hand-set pipeline settings, on 1998-2024 Senate races. Score: error of the poll average against the result."""
import sys;sys.path.insert(0,'models')
import numpy as np,pandas as pd
from core import DIV
H=pd.read_csv('data/processed/history_base.csv');H=H[H.dvr];P=pd.read_csv('data/processed/polls_hist.csv');P['part']=P.part.fillna('');P['N']=P.N.fillna(600).clip(100,3000)
P=P.merge(H[['year','st','special','margin']],on=['year','st','special'])
P['pc']=P.pollster.astype(str).str.replace(r'\s*\((D|R)\)','',regex=True)
def score(tau=14,win=60,pd_=.5,nexp=.5,rep=.5,adj=True,dA=2.12,rA=-2.15,close=20):
    Q=P[P.age<=win].copy();w=np.exp(-Q.age/tau)*(Q.N/600)**nexp*np.where(Q.part!='',pd_,1.)
    Q['cnt']=Q.groupby(['year','st','special','pc']).pc.transform('size');Q['w']=w/Q.cnt**rep
    Q['mm']=Q.m-(np.where(Q.part=='D',dA,np.where(Q.part=='R',rA,0.)) if adj else 0)
    a=Q.groupby(['year','st','special']).apply(lambda g:pd.Series(dict(p=np.sum(g.w*g.mm)/g.w.sum(),y=g.margin.iloc[0])),include_groups=False).reset_index()
    a=a[a.p.abs()<close];e=a.y-a.p;per=(e**2).groupby(a.year).mean()
    return np.sqrt((e**2).mean()),((np.sign(a.p)==np.sign(a.y)).mean()),len(a)
base=score();print('BASE rmse %.3f call %.3f n=%d'%base)
for nm,kw in [('tau=3',dict(tau=3)),('tau=7',dict(tau=7)),('tau=14',dict(tau=14)),('tau=21',dict(tau=21)),('tau=28',dict(tau=28)),('tau=60',dict(tau=60)),
 ('window=21',dict(win=21)),('window=30',dict(win=30)),('window=60',dict(win=60)),('window=90',dict(win=90)),
 ('sponsor wt 1',dict(pd_=1.)),('sponsor wt .5',dict(pd_=.5)),('sponsor wt .25',dict(pd_=.25)),('no sponsor adj',dict(adj=False)),
 ('N exp 0',dict(nexp=0.)),('N exp .5',dict(nexp=.5)),('N exp 1',dict(nexp=1.)),('repeat 0',dict(rep=0.)),('repeat .5',dict(rep=.5)),('repeat 1',dict(rep=1.))]:
    r=score(**kw);print(f'{nm:16s} rmse {r[0]:.3f} call {r[1]:.3f} n={r[2]}')
# Lean weighting: within-cycle R2 of margin on lean for alternative weights
piv=pd.read_csv('data/processed/pres_relmargin.csv',index_col=0);piv.columns=piv.columns.astype(int)
H2=H.copy();
for wl in (1.,.75,.5,.25):
    v=[]
    for r in H2.itertuples():
        pr=[a for a in piv.columns if a<r.year][-2:];v.append(wl*piv.loc[r.st,pr[-1]]+(1-wl)*piv.loc[r.st,pr[-2]])
    H2['L']=v;yd=H2.margin-H2.groupby('year').margin.transform('mean');xd=H2.L-H2.groupby('year').L.transform('mean');print('lean weight %.2f on latest: within-cycle r=%.4f'%(wl,np.corrcoef(xd,yd)[0,1]))
# Residual tails: justify t degrees of freedom
from scipy import stats
B=pd.read_csv('data/processed/backtest_final.csv');z=((B.margin-B.mu)/B.sd).dropna();print('resid t-fit df %.1f scale %.3f'%stats.t.fit(z,floc=0)[0:1]+(stats.t.fit(z,floc=0)[2],) if False else 'resid t-fit',np.round(stats.t.fit(z,floc=0),3))
by=(B[B.p.notna()&(B.mu.abs()<12)].assign(e=lambda d:d.margin-d.mu)).groupby('year').e.mean();print('cycle-bias SD %.2f; kurtosis %.2f'%(by.std(),stats.kurtosis(by)))
