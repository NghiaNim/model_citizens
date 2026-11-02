import sys;sys.path.insert(0,'models')
import numpy as np,pandas as pd,json,re
from scipy import stats,optimize
from core import *
rng=np.random.default_rng(20261103)
fit_=json.load(open('data/processed/full_fit.json')); th=unpack(np.array(fit_['theta']))
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def expit(x): return 1/(1+np.exp(-x))
# --- margin expansion g(mu), estimated on close races with cycle-clustered SE (beta 1.13, se .06 for |mu|<10)
BETA=.13
def g(mu): return mu*(1+BETA*np.exp(-(mu/20.)**2))
# Nested re-calibration AFTER g
B=pd.read_csv('data/processed/backtest_fusion.csv'); B['pg']=stats.norm.cdf(g(B.mu)/B.sd); x=logit(B.pg.values); y=(B.margin>0).values.astype(float)
def nl(a,x,y): pr=np.clip(expit(a*x),1e-6,1-1e-6); return -np.sum(y*np.log(pr)+(1-y)*np.log(1-pr))
As=[];B['pc']=np.nan
for yr in B.year.unique():
    tr=(B.year!=yr).values; a=optimize.minimize_scalar(nl,bounds=(.5,3),args=(x[tr],y[tr]),method='bounded').x; As.append(a); B.loc[~tr,'pc']=expit(a*x[~tr])
K=float(np.mean(As)); print('post-g recalibration slope k=%.3f (range %.2f-%.2f)'%(K,min(As),max(As)))
def br(p,y): return np.mean((p-y)**2)
def ll(p,y): p=np.clip(p,.003,.997); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
print('LOCO with g+k: Brier %.4f LL %.4f'%(br(B.pc,y),ll(B.pc,y)))
B['fav']=np.maximum(B.pc,1-B.pc);B['won']=((B.pc>.5)==(B.margin>0)).astype(int)
cal=B.groupby(pd.cut(B.fav,[.5,.6,.7,.8,.9,.95,.99,.999,1.001],right=False),observed=True).agg(n=('won','size'),pred=('fav','mean'),obs=('won','mean')).round(3);print(cal.to_string());cal.to_csv('output/calibration_backtest.csv')
B.to_csv('data/processed/backtest_final.csv',index=False)
# --- 2026 inputs
D=pd.read_csv('data/processed/race_2026_base.csv'); ENV=4.0
D['f']=D.f0+ENV
# independent-candidate analog correction: polls vs result  KS14 -12.1, UT22 -3.0, NE24 -4.3 (last-30d averages)
I_ERR=np.array([-12.1,-3.0,-4.3]); I_SHIFT=float(I_ERR.mean()*3/(3+1)); I_SD=4.5; print('independent shift (n=3, shrunk 25%% to 0): %.2f'%I_SHIFT)
D.loc[D.kind=='I','f']=D.f0+ENV-0   # fundamentals for independents kept but fscale widens
mu,v,w=mean_var(D,th); D['w_poll']=w; D['mu_raw']=mu; D['var']=v
D['mu']=g(mu); ind=(D.kind=='I').values; D.loc[ind,'mu']=D.mu[ind]+I_SHIFT
D['sd_marg']=np.sqrt(marginal_sd(D,th,v)**2+ind*I_SD**2)
# --- correlated Monte Carlo
S=400000; n=len(D); nat=stats.t.rvs(8,size=S,random_state=rng)*th['t_n']/np.sqrt(8/6)
lf=rng.normal(0,th['t_l'],S); divs=np.unique(D['div']); dv={d:rng.normal(0,th['t_r'],S) for d in divs}
idio=stats.t.rvs(12,size=(S,n),random_state=rng)/np.sqrt(12/10)
sim=np.zeros((S,n)); shared_i=rng.normal(0,I_SD,S)   # independent-candidate overstatement is a shared polling artefact
for i,r in enumerate(D.itertuples()):
    sim[:,i]=r.mu+nat+lf*r.z+dv[getattr(r,'_asdict')()['div']]+np.sqrt(r.var)*idio[:,i]+(shared_i if r.kind=='I' else 0)
pm=np.mean(sim>0,axis=0); D['p_mc']=pm
D['P_model']=expit(K*logit(np.clip(pm,1/S,1-1/S)))
# shift each column so marginals equal calibrated P (copula-preserving) for joint distribution
sim_c=sim.copy()
for i in range(n):
    q=np.quantile(sim[:,i],1-D.P_model.iloc[i]); sim_c[:,i]=sim[:,i]-q
np.save('data/processed/sim_calibrated.npy',sim_c[:60000].astype('float32'))
# --- Ratings model
Rt=pd.read_csv('data/processed/ratings_wiki_raw.csv'); b=json.load(open('data/processed/ratings_fit.json'))['rating_b']
def sc(x):
    x=str(x).lower().replace('(flip)','').strip()
    if 'toss' in x: return 0.
    v={'safe':4,'solid':4,'likely':3,'lean':2,'tilt':1}.get(x.split()[0]);
    if v is None: return np.nan
    return float(v) if re.search(r'\b(d|dfl|i)\b',x) else float(-v)
Rt['s']=Rt.rating.map(sc); sr=Rt.groupby('st').s.agg(['mean','std','count']); D=D.merge(sr.rename(columns={'mean':'rate_score','std':'rate_sd','count':'rate_n'}),left_on='st',right_index=True,how='left')
D['P_ratings']=expit(b*D.rate_score)
# --- Markets
ka=pd.DataFrame(json.load(open('data/raw/markets/kalshi_snapshot.json')));po=json.load(open('data/raw/markets/polymarket_snapshot.json'))
import urllib.request
KX={'TX':[('James Talarico',.73,.74),('Ken Paxton',.26,.27)],'OK':[('Kevin Hern',.977,.989),('N Thomas',.015,.023)],'TN':[('Bill Hagerty',.978,.984),('Marquita Bradshaw',.017,.019)],'NE':[('Pete Ricketts',.69,.70),('Dan Osborn',.30,.31)],'KY':[('Andy Barr',.958,.959),('Charles Booker',.041,.043)]}
def kal(st,cand):
    if st=='LA': return np.nan   # Kalshi's SENATELA-26 ticker actually lists the Kentucky race; Louisiana not on feed
    rows=[(a,x,c) for a,x,c in KX.get(st,[])]
    if not rows:
        g_=ka[ka.st==st]; rows=[(r.title,float(r.bid),float(r.ask)) for r in g_.itertuples()]
    tot=sum((x+c)/2 for _,x,c in rows)
    for a,x,c in rows:
        if cand.lower() in a.lower().replace('’',"'") or cand.lower().split('-')[-1] in a.lower(): return ((x+c)/2)/tot
    return np.nan
def pol(st,cand):
    rows=[r for r in po.get(st,[]) if not r['closed'] and r['ev'].lower().startswith({'AK':'alaska','AL':'alabama','AR':'arkansas','CO':'colorado','DE':'delaware','FL':'florida','GA':'georgia','IA':'iowa','ID':'idaho','IL':'illinois','KS':'kansas','KY':'kentucky','LA':'louisiana','MA':'massachusetts','ME':'maine','MI':'michigan','MN':'minnesota','MS':'mississippi','MT':'montana','NC':'north carolina','NE':'nebraska','NH':'new hampshire','NJ':'new jersey','NM':'new mexico','OH':'ohio','OK':'oklahoma','OR':'oregon','RI':'rhode island','SC':'south carolina','SD':'south dakota','TN':'tennessee','TX':'texas','VA':'virginia','WV':'west virginia','WY':'wyoming'}[st])]
    rows=sorted(rows,key=lambda r:-(r['vol'] or 0))
    for r in rows:
        q=r['q'].lower()
        if cand.lower() in q or (cand in('Democrat',) and q.startswith('democrat')): return r['yes']
    return np.nan
WIN={'Lujan':'Luj'}
recs=[]
for r in D.itertuples():
    cd={'Lujan':'Luján'}.get(r.dname,r.dname); key=cd
    k=kal(r.st,key); p=pol(r.st,key)
    if np.isnan(p) and r.st=='MT': p=[x for x in po['MT'] if x['q']=='Independent'][0]['yes']
    if np.isnan(p) and r.st=='NE': p=[x for x in po['NE'] if x['q']=='Independent'][0]['yes']
    if np.isnan(p) and r.st=='DE': p=[x for x in po['DE'] if x['q']=='Democrat'][0]['yes']
    recs.append(dict(st=r.st,kalshi=k,poly=p))
M=pd.DataFrame(recs); M['P_market']=M[['kalshi','poly']].mean(axis=1); D=D.merge(M,on='st')
# --- user's predictions
U=pd.read_csv('predictions.csv'); U['st']=U.race_id.str[:2]; U['pd_user']=np.where(U.predicted_winner_party.isin(['Democratic']),U.win_probability/100,1-U.win_probability/100)
D=D.merge(U[['st','predicted_winner_party','candidate_name','win_probability','pd_user']],on='st')
# non-R party mapped to "challenger side": for I races P_* is the independent's probability
D['P_evid']=expit((logit(D.P_model)+logit(D.P_ratings))/2)   # no market: independent of user's source
D['P_pool']=expit((logit(D.P_model)+logit(D.P_ratings)+logit(D.P_market))/3)
for c in ['P_model','P_ratings','P_market','P_evid','P_pool']: D['fav_'+c]=np.where(D.pd_user>=.5,D[c],1-D[c])
cols=['st','candidate_name','win_probability','fav_P_model','fav_P_ratings','fav_P_market','fav_P_evid','fav_P_pool','p','n30','neff','und','w_poll','mu','sd_marg','rate_score']
pd.set_option('display.width',220)
print(D[cols].round({'fav_P_model':3,'fav_P_ratings':3,'fav_P_market':3,'fav_P_evid':3,'fav_P_pool':3,'p':1,'neff':1,'und':1,'w_poll':2,'mu':1,'sd_marg':1,'rate_score':2}).sort_values('win_probability').to_string())
D.to_csv('output/race_table_full.csv',index=False)
print('corr user vs market (logit, non-100):',np.corrcoef(logit(D[D.win_probability<100].pd_user),logit(D[D.win_probability<100].P_market))[0,1],' mean abs pts:',(D[D.win_probability<100].pd_user-D[D.win_probability<100].P_market).abs().mean()*100)
# --- seats
winners=sim_c>0; dside=(D.kind=='D').values
Dseats=34+winners[:,dside].sum(axis=1); Iw=winners[:,~dside].sum(axis=1)
print('Democratic seats: mean %.1f median %d 80%%CI %s  P(D>=51)=%.3f P(D+I>=51)=%.3f'%(Dseats.mean(),np.median(Dseats),np.quantile(Dseats,[.1,.9]),np.mean(Dseats>=51),np.mean(Dseats+Iw>=51)))
print('Seat dist',{int(k):round(float(v),3) for k,v in pd.Series(Dseats).value_counts(normalize=True).sort_index().items() if v>.01})
cor=pd.DataFrame(winners,columns=D.st)[['TX','OH','IA','KS','ME','MI','AK','NE','SC']].corr().round(2);print(cor.to_string())
pd.DataFrame(dict(seats=Dseats)).value_counts(normalize=True).sort_index().to_csv('output/seat_distribution.csv')
