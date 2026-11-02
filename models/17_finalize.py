"""Final numbers: calibration from Senate+governor data, no margin expansion. Rewrites output tables the report reads.
The previous Senate-only version is preserved as output/final_comparison_senate_only.csv."""
import sys;sys.path.insert(0,'models')
import numpy as np,pandas as pd,json,os,shutil
from scipy import stats
from core import *
def logit(p):p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
def expit(x):return 1/(1+np.exp(-x))
if not os.path.exists('output/final_comparison_senate_only.csv'):shutil.copy('output/final_comparison.csv','output/final_comparison_senate_only.csv')
O=pd.read_csv('output/final_comparison_senate_only.csv');D=pd.read_csv('output/race_table_full.csv');R=pd.read_csv('output/revised_decisions.csv').set_index('race')
th=unpack(np.array(json.load(open('data/processed/full_fit.json'))['theta']));ind=(D.kind=='I').values;ISH=-4.85
KC,KG,KS=1.051,.795,1.504
mg=np.where(ind,D.mu_raw+ISH,D.mu_raw);sd=O.sd.values;sgn=(D.pd_user>=.5).values
Fx=lambda c:np.where(sgn,c,1-c)
def Pm(k,sh=0.):return expit(np.where(ind,1.,k)*logit(stats.norm.cdf((mg+sh)/sd)))
model=Pm(KC);a,b=Fx(Pm(KG)),Fx(Pm(KS))
O['model']=Fx(model);O['lo']=np.minimum(a,b);O['hi']=np.maximum(a,b)
pool=lambda m:expit((logit(m)+logit(D.P_ratings.values)+logit(D.P_market.values))/3)
O['pool']=Fx(pool(model));O['margin']=np.round(mg,1)
O['proposed']=O.race.map(R.revised);O['decision']=O.race.map(R.why)
O['R53']=Fx(Pm(KC,-5.3));O['R3']=Fx(Pm(KC,-3));O['D2']=Fx(Pm(KC,2))
O.to_csv('output/final_comparison.csv',index=False)
# joint
def mcsim(S=400000,seed=21,mult=1.):
    r=np.random.default_rng(seed);nat=stats.t.rvs(8,size=S,random_state=r)*th['t_n']*mult/np.sqrt(8/6);lf=r.normal(0,th['t_l'],S);dv={d:r.normal(0,th['t_r'],S) for d in np.unique(D['div'])};idio=stats.t.rvs(12,size=(S,len(D)),random_state=r)/np.sqrt(12/10);si=r.normal(0,4.5,S);s=np.zeros((S,len(D)))
    for i,(z,dd,ki,v_) in enumerate(zip(D.z,D['div'],D.kind,D['var'])):s[:,i]=mg[i]+nat+lf*z+dv[dd]+np.sqrt(v_)*idio[:,i]+(si if ki=='I' else 0)
    return s
S0=mcsim();ds=(D.kind=='D').values
def joint(m):
    s=S0.copy()
    for i in range(len(D)):s[:,i]-=np.quantile(s[:,i],1-np.clip(m[i],1e-4,1-1e-4))
    return 34+(s[:,ds]>0).sum(1)
pdrev=np.where(sgn,O.proposed/100,1-O.proposed/100);OUT={}
for nm,m in (('model',model),('pool',pool(model)),('market',D.P_market.values),('user',D.pd_user.values),('proposed',pdrev)):
    d=joint(m);OUT[nm]=dict(mean=float(d.mean()),med=int(np.median(d)),q10=float(np.quantile(d,.1)),q90=float(np.quantile(d,.9)),p51=float((d>=51).mean()),p50=float((d==50).mean()));print(nm,{k:round(v,3) for k,v in OUT[nm].items()})
    if nm=='proposed':pd.Series(d).value_counts(normalize=True).sort_index().rename('p').to_csv('output/seat_distribution.csv')
json.dump(OUT,open('output/seat_summary.json','w'),indent=1)
Up=O.user/100;Pr=O.proposed/100;eb=lambda f,t:np.sum(t*(1-f)**2+(1-t)*f**2);el=lambda f,t:(lambda f:-np.sum(t*np.log(f)+(1-t)*np.log(1-f)))(np.clip(f,.005,.995))
ES=pd.DataFrame([dict(truth=n,brier_user=eb(Up,t),brier_prop=eb(Pr,t),logl_user=el(Up,t),logl_prop=el(Pr,t)) for n,t in (('model',O.model),('ratings',O.ratings),('market',O.market),('pool',O.pool))]).round(3);ES.to_csv('output/expected_score.csv',index=False);print(ES.to_string())
pd.set_option('display.width',200);print(O[O.user<100][['race','user','proposed','model','lo','hi','ratings','market','pool','margin','sd','R53','D2']].round(3).sort_values('user').to_string());print(O[(O.user==100)][['race','proposed','model','ratings','market','pool','R53']].round(3).sort_values('pool').head(12).to_string())
B=pd.read_csv('data/processed/backtest_final.csv');G=pd.read_csv('data/processed/governor_replication.csv')
ps=expit(KC*logit(stats.norm.cdf(B.mu/B.sd)));y=(B.margin>0)*1.;fav=np.maximum(ps,1-ps);won=((ps>.5)==(B.margin>0))*1
print('Senate backtest at combined k: Brier %.4f LL %.4f call %.3f'%(np.mean((ps-y)**2),-np.mean(y*np.log(ps.clip(.003,.997))+(1-y)*np.log(1-ps.clip(.003,.997))),won.mean()))
c=pd.DataFrame(dict(fav=fav,won=won));Cal=c.groupby(pd.cut(c.fav,[.5,.6,.7,.8,.9,.95,.99,.999,1.001],right=False),observed=True).agg(n=('won','size'),pred=('fav','mean'),obs=('won','mean')).round(3).reset_index();Cal['fav']=Cal.fav.astype(str);print(Cal.to_string());Cal.to_csv('output/calibration_backtest.csv',index=False)
sdG=np.sqrt(3.15**2+3.62**2/np.maximum(G.neff,.2)+(.15*G.p)**2+2.43**2);pg=expit(KC*logit(stats.norm.cdf(G.p/sdG)));print('Gov at combined k: Brier %.4f; LL %.4f'%(np.mean((pg-G.y)**2),-np.mean(G.y*np.log(pg.clip(.003,.997))+(1-G.y)*np.log(1-pg.clip(.003,.997)))))
a=c[(c.fav>=.5)&(c.fav<.9)];print('mid',len(a),a.fav.mean().round(3),a.won.mean().round(3));a=c[(c.fav>=.9)&(c.fav<.99)];print('hi',len(a),a.fav.mean().round(3),a.won.mean().round(3))
