import numpy as np,pandas as pd,json,urllib.request,urllib.parse
from scipy import stats,optimize
B=pd.read_csv('data/processed/backtest_final.csv'); rng=np.random.default_rng(7)
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def expit(x): return 1/(1+np.exp(-x))
g=lambda m:m*(1+.13*np.exp(-(m/20.)**2))
B['mg']=g(B.mu); B['err']=B.margin-B.mg; B['ld']=np.sign(B.mg); B['leadwon']=(np.sign(B.margin)==B.ld).astype(int); B['pl']=np.maximum(B.pc,1-B.pc)
P=B[B.p.notna()].copy()
def cboot(df,fn,B_=4000):
    ys=df.year.unique(); out=[]
    grp={y:df[df.year==y] for y in ys}
    for _ in range(B_):
        s=pd.concat([grp[y] for y in rng.choice(ys,len(ys))]); 
        if len(s): out.append(fn(s))
    return np.nanquantile(out,[.05,.95])
def report(name,df):
    if not len(df): print(name,'empty');return
    ci=cboot(df,lambda s:s.leadwon.mean())
    print(f"{name:58s} n={len(df):3d} cycles={df.year.nunique():2d} leader won={df.leadwon.mean():.3f} [90%CI {ci[0]:.2f},{ci[1]:.2f}] model said={df.pl.mean():.3f} | mean err toward leader={np.mean(df.err*df.ld):+.2f} | miss={list((df[df.leadwon==0].year.astype(str).str[2:]+df[df.leadwon==0].st))[:12]}")
print('=== reference classes (LOCO, leader by model margin 2<|mu|<8 unless stated)')
c=P[(P.mg.abs()>=2)&(P.mg.abs()<8)]
report('all, 2-8',c)
report('2-5',P[(P.mg.abs()>=2)&(P.mg.abs()<5)]); report('5-8',P[(P.mg.abs()>=5)&(P.mg.abs()<8)]); report('8-12',P[(P.mg.abs()>=8)&(P.mg.abs()<12)])
report('D leads in red state (lean<=-8): 2026 TX/OH/IA/KS/AK class',c[(c.ld>0)&(c.lean<=-8)])
report('D leads in red state, midterms only',c[(c.ld>0)&(c.lean<=-8)&(c.midterm)])
report('D leads in red state, pres years only',c[(c.ld>0)&(c.lean<=-8)&(~c.midterm)])
report('D leads, any state, midterm',c[(c.ld>0)&(c.midterm)])
report('D leads, opposition(D)-wave midterms 2006/2018',c[(c.ld>0)&(c.year.isin([2006,2018]))])
report('Challenger leads incumbent (ME/OH/KS/AK class)',c[(c.inc!=0)&(np.sign(c.inc)!=c.ld)])
report('Challenger leads incumbent, midterms',c[(c.inc!=0)&(np.sign(c.inc)!=c.ld)&(c.midterm)])
report('R incumbent trailing D, red/purple',c[(c.inc==-1)&(c.ld>0)])
report('Incumbent trailing, cross-partisan state (R inc lean>0/D inc lean<0)',P[(P.mg.abs()<8)&(((P.inc==-1)&(P.lean>0)&(P.ld>0))|((P.inc==1)&(P.lean<0)&(P.ld<0)))])
report('R leads narrowly in red state (SC/MS/LA/FL class), lean<=-8',P[(P.mg.abs()>=2)&(P.mg.abs()<10)&(P.ld<0)&(P.lean<=-8)])
report('open seats',c[c.inc==0])
# residual tilt: err on z in close races
import statsmodels.api as sm
cl=P[P.mg.abs()<12]; X=sm.add_constant(pd.DataFrame(dict(z=cl.z,inc=cl.inc,mid=cl.midterm.astype(float),pres_party=cl.president*cl.midterm)))
m=sm.OLS(cl.err,X).fit(cov_type='cluster',cov_kwds={'groups':cl.year}); print(m.summary().tables[1])
by=cl.groupby('year').err.mean().round(2);print('close-race bias by cycle',by.to_dict(),' sd=%.2f'%by.std(),' midterm sd=%.2f pres sd=%.2f'%(by[[y for y in by.index if y%4==2]].std(),by[[y for y in by.index if y%4==0]].std()),' midterm mean=%.2f pres mean=%.2f'%(by[[y for y in by.index if y%4==2]].mean(),by[[y for y in by.index if y%4==0]].mean()))
# bootstrap k (cycle) : uncertainty of sharpening
x=logit(B.pg.values);y=(B.margin>0).values.astype(float)
def nl(a,x,y): pr=np.clip(expit(a*x),1e-6,1-1e-6); return -np.sum(y*np.log(pr)+(1-y)*np.log(1-pr))
ks=[]
for _ in range(600):
    ys=rng.choice(B.year.unique(),B.year.nunique()); ii=np.concatenate([np.where(B.year==v)[0] for v in ys])
    ks.append(optimize.minimize_scalar(nl,bounds=(.5,4),args=(x[ii],y[ii]),method='bounded').x)
print('k cycle-bootstrap: mean %.2f, 5%% %.2f, 95%% %.2f'%(np.mean(ks),*np.quantile(ks,[.05,.95])))
np.save('data/processed/k_boot.npy',np.array(ks))
# Market control price
H={'User-Agent':'Mozilla/5.0'}
u='https://gamma-api.polymarket.com/public-search?'+urllib.parse.urlencode(dict(q='Which party will win the Senate in 2026',limit_per_type=5,events_status='active'))
d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=30))
for e in d['events'][:3]:
    print('POLY',e['title'],[(m.get('groupItemTitle'),m.get('outcomePrices')) for m in e['markets'][:4]])
try:
    for t in ['CONTROLS-2026','KXCONTROLS-2026','SENATE-26','CONTROLSENATE-26']:
        r=json.load(urllib.request.urlopen(urllib.request.Request(f'https://api.elections.kalshi.com/trade-api/v2/markets?event_ticker={t}&limit=10',headers=H),timeout=20))
        if r['markets']: print('KALSHI',t,[(m.get('yes_sub_title'),m.get('yes_bid_dollars'),m.get('yes_ask_dollars')) for m in r['markets']])
except Exception as e: print(e)
