"""Re-derive the three-source average using calibration informed by Senate+governor data, no margin expansion (it failed to replicate)."""
import numpy as np,pandas as pd
from scipy import stats,optimize
def logit(p):p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
def expit(x):return 1/(1+np.exp(-x))
def nl(a,x,y):pr=np.clip(expit(a*x),1e-6,1-1e-6);return -np.sum(y*np.log(pr)+(1-y)*np.log(1-pr))
B=pd.read_csv('data/processed/backtest_final.csv');G=pd.read_csv('data/processed/governor_replication.csv')
yS=(B.margin>0).values*1.;xS=logit(stats.norm.cdf(B.mu/B.sd))         # Senate, no expansion
sdG=np.sqrt(3.15**2+3.62**2/np.maximum(G.neff,.2)+(.15*G.p)**2+2.43**2);xG=logit(stats.norm.cdf(G.p/sdG));yG=G.y.values
fit=lambda x,y:optimize.minimize_scalar(nl,bounds=(.3,4),args=(x,y),method='bounded').x
kS,kG,kA=fit(xS,yS),fit(xG,yG),fit(np.r_[xS,xG],np.r_[yS,yG])
print('k (no expansion): Senate %.3f, Governor %.3f, combined %.3f'%(kS,kG,kA))
# competitive subset only: |x| not extreme
def sub(x,y):m=(np.abs(x)<logit(.99));return x[m],y[m]
print('competitive-only k: Senate %.3f, Governor %.3f, combined %.3f'%(fit(*sub(xS,yS)),fit(*sub(xG,yG)),fit(*sub(np.r_[xS,xG],np.r_[yS,yG]))))
print('share of competitive races that are governors',(np.abs(xG)<logit(.99)).sum(),(np.abs(xS)<logit(.99)).sum())
O=pd.read_csv('output/final_comparison.csv');D=pd.read_csv('output/race_table_full.csv')
ind=(D.kind=='I').values;ISH=-4.85
mu=D.mu_raw.values
mg=np.where(ind,mu+ISH,mu)    # no expansion
sd=O.sd.values
pm=stats.norm.cdf(mg/sd)
def model(k):return expit(np.where(ind,1.,k)*logit(pm))
sgn=(D.pd_user>=.5).values
rows={}
for nm,k in (('k=1 (no sharpening)',1.),('combined k',fit(*sub(np.r_[xS,xG],np.r_[yS,yG])))):
    M=model(k);pool=expit((logit(M)+logit(D.P_ratings)+logit(D.P_market))/3)
    rows[nm]=np.where(sgn,pool,1-pool);rows[nm+' model']=np.where(sgn,M,1-M)
T=pd.DataFrame(rows);T.insert(0,'race',O.race);T.insert(1,'orig',O.user);T.insert(2,'adopted',O.proposed);T.insert(3,'ratings',O.ratings);T.insert(4,'market',O.market)
pd.set_option('display.width',200);print(T[T.orig<100].round(3).sort_values('orig').to_string());print(T[T.orig==100].sort_values('k=1 (no sharpening)').head(10).round(3).to_string())
T.to_csv('output/revised_after_replication.csv',index=False)
# empirical pooled classes
Bp=B[B.p.notna()];Sl=lambda a,b:Bp[(Bp.mu.abs()>=a)&(Bp.mu.abs()<b)];Gl=lambda a,b:G[(G.p.abs()>=a)&(G.p.abs()<b)]
for a,b in [(0,2),(2,4),(4,6),(6,8),(8,12)]:
    s,g=Sl(a,b),Gl(a,b);ws=(np.sign(s.margin)==np.sign(s.mu)).sum();wg=(np.sign(g.act)==np.sign(g.p)).sum()
    print(f'raw lead {a}-{b}: Senate {ws}/{len(s)}={ws/len(s):.2f}; Governor {wg}/{len(g)}={wg/len(g):.2f}; pooled {(ws+wg)/(len(s)+len(g)):.2f}')
