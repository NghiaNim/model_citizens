"""(a) empirical error of close races; (b) nested Platt recalibration; (c) 538 benchmark; (d) ratings ordinal model; (e) tail/upset rates."""
import numpy as np,pandas as pd,re,json
from io import StringIO
from bs4 import BeautifulSoup
from scipy import stats,optimize
R=pd.read_csv('data/processed/backtest_fusion.csv')
R['err']=R.margin-R.mu
for lo,hi in [(0,3),(3,6),(6,10),(10,15),(15,25),(25,100)]:
    g=R[(R.mu.abs()>=lo)&(R.mu.abs()<hi)&R.p.notna()]
    print(f"|mu| {lo}-{hi}: n={len(g)} RMSE={np.sqrt((g.err**2).mean()):.2f} model_sd={g.sd.mean():.2f} leader_won={np.mean((g.mu>0)==(g.margin>0)):.3f} pred={np.mean(np.maximum(g.pd,1-g.pd)):.3f}")
g=R[R.p.notna()&(R.mu.abs()<15)]; print('close races residual sd by era:', g.groupby(pd.cut(g.year,[1997,2008,2016,2024]),observed=True).err.agg(['size','std','mean']).round(2).to_string())
# nested Platt: logit(p)->a*logit(p) fit on other cycles
def logit(p): p=np.clip(p,1e-5,1-1e-5); return np.log(p/(1-p))
def nllP(a,x,y): pr=1/(1+np.exp(-a*x)); pr=np.clip(pr,1e-6,1-1e-6); return -np.sum(y*np.log(pr)+(1-y)*np.log(1-pr))
x=logit(R.pd.values); y=(R.margin>0).values.astype(float); R['pcal']=np.nan; As=[]
for yr in R.year.unique():
    tr=R.year!=yr; m=(np.abs(x)<logit(.9995))&tr
    a=optimize.minimize_scalar(nllP,bounds=(.5,3),args=(x[m],y[m]),method='bounded').x; As.append(a)
    R.loc[~tr,'pcal']=1/(1+np.exp(-a*x[~tr]))
print('Platt slope by held-out cycle',np.round(As,2),'mean',np.mean(As))
def br(p,y): return np.mean((p-y)**2)
def ll(p,y): p=np.clip(p,.003,.997); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
yy=(R.margin>0).astype(float)
print('Raw   Brier %.4f LL %.4f | Recal Brier %.4f LL %.4f'%(br(R.pd,yy),ll(R.pd,yy),br(R.pcal,yy),ll(R.pcal,yy)))
R['fav']=np.maximum(R.pcal,1-R.pcal);R['won']=((R.pcal>.5)==(R.margin>0)).astype(int)
print(R.groupby(pd.cut(R.fav,[.5,.6,.7,.8,.9,.95,.99,.999,1.001],right=False),observed=True).agg(n=('won','size'),pred=('fav','mean'),obs=('won','mean')).round(3).to_string())
json.dump(dict(platt=float(np.mean(As))),open('data/processed/platt.json','w'))
# ------------ 538 benchmark
C=pd.read_csv('data/raw/fte/co_us_senate_elections.csv'); C=C[C.party.isin(['D','R'])].copy()
print(C.forecast_type.value_counts().to_dict(),C.year.unique())
C['forecast_date']=pd.to_datetime(C.forecast_date); C['election_date']=pd.to_datetime(C.election_date)
last=C[C.forecast_type=='deluxe'].sort_values('forecast_date').groupby(['year','state','special','party']).tail(1)
if last.empty: last=C[C.forecast_type=='classic'].sort_values('forecast_date').groupby(['year','state','special','party']).tail(1)
print('538 final-forecast dates',last.groupby('year').forecast_date.max().to_dict())
d=last[last.party=='D'].copy(); d['special']=d.special.astype(str).str.lower().eq('true')
m=R.merge(d[['year','state','special','probwin','probwin_outcome']],left_on=['year','st','special'],right_on=['year','state','special'])
ym=m.probwin_outcome.astype(float)
print('538 deluxe vs mine on same n=%d races | 538 Brier %.4f LL %.4f | mine raw Brier %.4f LL %.4f | mine recal Brier %.4f LL %.4f'%(len(m),br(m.probwin,ym),ll(m.probwin,ym),br(m.pd,ym),ll(m.pd,ym),br(m.pcal,ym),ll(m.pcal,ym)))
k=last.copy();k['fav']=k.probwin
f=k[k.fav>=.5]
print('538 reliability (favourite):\n',f.groupby(pd.cut(f.fav,[.5,.6,.7,.8,.9,.95,.99,.995,.999,1.001],right=False),observed=True).agg(n=('fav','size'),pred=('fav','mean'),obs=('probwin_outcome','mean')).round(3).to_string())
print('538 >=.99 upsets:',f[(f.fav>=.99)&(f.probwin_outcome==0)][['year','state','candidate','fav']].to_string())
# ------------ Ratings ordinal: 2010-2024 Wikipedia tables
SC={'safe':4,'solid':4,'likely':3,'lean':2,'tilt':1,'tossup':0,'toss-up':0,'toss':0}
def score(x):
    x=str(x).lower().strip()
    if 'toss' in x: return 0.
    for k,v in SC.items():
        if x.startswith(k):
            if re.search(r'\b(d|dfl)\b',x.replace('(flip)','')): return float(v)
            if re.search(r'\b(r)\b',x.replace('(flip)','')): return float(-v)
            if re.search(r'\bi\b',x): return float(v)
    return np.nan
names=dict(zip("Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan Minnesota Mississippi Missouri Montana Nebraska Nevada New_Hampshire New_Jersey New_Mexico New_York North_Carolina North_Dakota Ohio Oklahoma Oregon Pennsylvania Rhode_Island South_Carolina South_Dakota Tennessee Texas Utah Vermont Virginia Washington West_Virginia Wisconsin Wyoming".replace('_','~').split(),"AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()))
names={k.replace('~',' '):v for k,v in names.items()}
Hb=pd.read_csv('data/processed/history_base.csv')
rr=[]
for y in [2010,2012,2014,2016,2018,2020,2022,2024]:
    s=BeautifulSoup(open(f'data/raw/wikihist/sen{y}.html',encoding='utf8'),'lxml')
    for x in s.select('sup.reference,span.sortkey,style'): x.decompose()
    for t in s.select('table.wikitable'):
        h=t.get_text(' ',strip=True)[:500]
        if not ('Cook' in h and ('Sabato' in h or 'Rothenberg' in h)): continue
        df=pd.read_html(StringIO(str(t)))[0]
        df.columns=[str(c[-1]) if isinstance(c,tuple) else str(c) for c in df.columns]
        rc=[c for c in df.columns if re.search(r'(Cook|IE|Rothenberg|Sabato|CBS|Politico|RCP|Fox|DDHQ|538|Econ|NYT|CNN|Daily Kos|ED|Inside|Crystal|Sab|Roll)',c)]
        for _,r in df.iterrows():
            nm=re.sub(r'\s*\(.*','',str(r.iloc[0])).strip()
            sp='special' in str(r.iloc[0]).lower() or 'class' in str(r.iloc[0]).lower()
            if nm not in names: continue
            sc=[score(r[c]) for c in rc]; sc=[v for v in sc if v==v]
            if sc: rr.append(dict(year=y,st=names[nm],rawname=r.iloc[0],sp=sp,score=np.mean(sc),n=len(sc)))
        break
RT=pd.DataFrame(rr); print(RT.groupby('year').size().to_dict())
Hh=Hb[Hb.year>=2010]
rows=[]
for _,r in RT.iterrows():
    g=Hh[(Hh.year==r.year)&(Hh.st==r.st)]
    if len(g)==0: continue
    if len(g)>1: g=g[g.special==r.sp] if (g.special==r.sp).any() else g
    g=g.iloc[0]
    if g.winparty not in('DEMOCRAT','REPUBLICAN'):continue
    rows.append(dict(year=r.year,st=r.st,score=r.score,n=r.n,win=int(g.winparty=='DEMOCRAT'),margin=g.margin))
RT=pd.DataFrame(rows).drop_duplicates(); RT.to_csv('data/processed/ratings_hist.csv',index=False); print('rating sample',len(RT))
def nllR(b,x,y,lamb=.02):
    pr=1/(1+np.exp(-b*x)); pr=np.clip(pr,1e-9,1-1e-9); return -np.sum(y*np.log(pr)+(1-y)*np.log(1-pr))
xs=RT.score.values;ys=RT.win.values;RT['pr']=np.nan
for yr in RT.year.unique():
    tr=RT.year!=yr; b=optimize.minimize_scalar(nllR,bounds=(.2,6),args=(xs[tr],ys[tr]),method='bounded').x; RT.loc[~tr,'pr']=1/(1+np.exp(-b*xs[~tr]))
b=optimize.minimize_scalar(nllR,bounds=(.2,6),args=(xs,ys),method='bounded').x
print('ratings logit slope',round(b,3),'LOCO Brier %.4f LL %.4f'%(br(RT.pr,RT.win),ll(RT.pr,RT.win)),'implied P at avg score 1,2,3,3.5:',np.round(1/(1+np.exp(-b*np.array([1,2,3,3.5]))),3))
RT['fav']=RT.score.abs(); RT['favwon']=((RT.score>0)==(RT.win==1)).astype(int)
print(RT[RT.score!=0].groupby(pd.cut(RT[RT.score!=0].fav,[0,.75,1.5,2.5,3.25,3.75,4.01]),observed=True).agg(n=('favwon','size'),obs=('favwon','mean')).round(3).to_string())
print('upsets with |score|>=3:',RT[(RT.fav>=3)&(RT.favwon==0)][['year','st','score','margin']].to_string())
json.dump(dict(rating_b=float(b)),open('data/processed/ratings_fit.json','w'))
mm=R.merge(RT,on=['year','st'],suffixes=('','_r'))
ym=(mm.margin>0).astype(float); print('Same races n=%d: ratings Brier %.4f LL %.4f; my raw %.4f %.4f, recal %.4f %.4f'%(len(mm),br(mm.pr,ym),ll(mm.pr,ym),br(mm.pd,ym),ll(mm.pd,ym),br(mm.pcal,ym),ll(mm.pcal,ym)))
# equal-weight logit pool of mine+ratings
pool=1/(1+np.exp(-(logit(mm.pcal)+logit(mm.pr))/2)); print('mine+ratings pool Brier %.4f LL %.4f'%(br(pool,ym),ll(pool,ym)))
R.to_csv('data/processed/backtest_fusion.csv',index=False)
