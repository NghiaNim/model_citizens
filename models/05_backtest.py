import sys;sys.path.insert(0,'models')
import numpy as np,pandas as pd,json
from scipy import stats
from core import *
H=pd.read_csv('data/processed/history_base.csv'); P=pd.read_csv('data/processed/polls_hist.csv')
P['part']=P.part.fillna('')
H=H[H.dvr].copy()
H['money']=H.logmoney.clip(-3,3).fillna(0); H['nomoney']=H.logmoney.isna().astype(float)
H['z']=H.lean/15.; H['div']=H.st.map(DIV); H['oppwind']=np.where(H.midterm,-H.president,0.); H['fscale']=1.
H.loc[H.year==1998,'inc']=0
adj=estimate_partisan_adj(P); print('Partisan poll house effects (D-sponsored, R-sponsored) vs race median:',np.round(adj,2))
ag=[]
for (y,st,sp),g in P.groupby(['year','st','special']):
    ag.append(dict(year=y,st=st,special=sp,**aggregate(g,adj)))
A=pd.DataFrame(ag); D=H.merge(A,on=['year','st','special'],how='left'); D['neff']=D.neff.fillna(0);D['npolls']=D.npolls.fillna(0)
print('races',len(D),'with polls',D.p.notna().sum())
# -------- coefficients of fundamentals, env estimate with nested leave-cycle-out
def prepare(D,test_year=None):
    D=D.copy(); yrs=sorted(D.year.unique()); D['f0']=np.nan; D['f']=np.nan; D['envsd']=np.nan
    for y in yrs:
        tr=D[(D.year!=y)&(D.year!=test_year)]; b=fund_fit(tr); idx=D.year==y
        f0=fund_raw(D[idx],b); D.loc[idx,'f0']=f0
        e,es=env_estimate(D.loc[idx,'p'].values,f0,D.loc[idx,'neff'].values)
        D.loc[idx,'f']=f0+e; D.loc[idx,'envsd']=es
    return D
years=sorted(D.year.unique())
def brier(p,y): return np.mean((p-y)**2)
def ll(p,y): p=np.clip(p,.003,.997); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
out=[];params=[]
def run_variant(name,mutate=lambda d:d,polls_only=False):
    recs=[]
    for y in years:
        d=mutate(prepare(D,y)); tr=d[d.year!=y]; te=d[d.year==y].copy()
        x,res=fit(tr,polls_only=polls_only); th=unpack(x)
        mu,v,w=mean_var(te,th,polls_only); sd=marginal_sd(te,th,v); 
        te['mu']=mu;te['sd']=sd;te['w']=w
        te['pd']=stats.norm.cdf(mu/sd)
        recs.append(te); params.append(dict(variant=name,year=y,**{k:round(float(v),3) for k,v in th.items()}))
    R=pd.concat(recs); yo=(R.margin>0).astype(float)
    print(f"{name:32s} RMSE {np.sqrt(np.mean((R.margin-R.mu)**2)):5.2f} | MAE {np.mean(np.abs(R.margin-R.mu)):5.2f} | Brier {brier(R.pd,yo):.4f} | logloss {ll(R.pd,yo):.4f} | call% {np.mean((R.pd>.5)==yo):.3f} ")
    c=R[(R.pd>.03)&(R.pd<.97)]; yc=(c.margin>0).astype(float)
    print(f"{'':32s} competitive n={len(c)} Brier {brier(c.pd,yc):.4f} logloss {ll(c.pd,yc):.4f}")
    return R
print('--- leave-one-cycle-out')
Rf=run_variant('C. fusion polls+fundamentals')
Rp=run_variant('A. polls only (polled races)',lambda d:d[d.p.notna()],polls_only=True)
Rfp=Rf[Rf.p.notna()]; yo=(Rfp.margin>0).astype(float); print(f"{'   fusion on same polled races':32s} RMSE {np.sqrt(np.mean((Rfp.margin-Rfp.mu)**2)):5.2f} | MAE {np.mean(np.abs(Rfp.margin-Rfp.mu)):5.2f} | Brier {brier(Rfp.pd,yo):.4f} | logloss {ll(Rfp.pd,yo):.4f}")
Rg=run_variant('B. fundamentals only',lambda d:d.assign(p=np.nan,und=0.,neff=0.,f=d.f0+0.))
Rf.to_csv('data/processed/backtest_fusion.csv',index=False)
pd.DataFrame(params).to_csv('data/processed/backtest_params.csv',index=False)
print(pd.DataFrame(params).query("variant.str.startswith('C')").drop(columns='variant').describe().loc[['mean','min','max']].round(3).to_string())
print('\nby cycle (fusion):'); 
for y,g in Rf.groupby('year'):
    yo=(g.margin>0).astype(float); print(y,'n',len(g),'bias(act-mu) %.2f'%(g.margin-g.mu).mean(),'RMSE %.2f'%np.sqrt(((g.margin-g.mu)**2).mean()),'Brier %.3f'%brier(g.pd,yo),'miss:',list(g[(g.pd>.5)!=(g.margin>0)].st))
# calibration
Rf['f_fav']=np.maximum(Rf.pd,1-Rf.pd); Rf['won']=((Rf.pd>.5)==(Rf.margin>0)).astype(int)
bins=[.5,.6,.7,.8,.9,.95,.99,.999,1.001]
print(Rf.groupby(pd.cut(Rf.f_fav,bins,right=False),observed=True).agg(n=('won','size'),pred=('f_fav','mean'),obs=('won','mean')).round(3).to_string())
Rf['zz']=(Rf.margin-Rf.mu)/Rf.sd; print('standardized residual sd %.3f kurt %.3f  t-fit:'%(Rf.zz.std(),stats.kurtosis(Rf.zz)),np.round(stats.t.fit(Rf.zz,floc=0),3))
print('coverage 80%%: %.3f  95%%: %.3f'%((Rf.zz.abs()<1.2816).mean(),(Rf.zz.abs()<1.96).mean()))
print('\nw vs neff:',Rf.groupby(pd.cut(Rf.neff,[-1,0,1,2,4,8,50]),observed=True).w.mean().round(2).to_dict())
d=prepare(D);x,res=fit(d);th=unpack(x);print('FULL fit',{k:round(float(v),3) for k,v in th.items()},'nll',round(res.fun,1))
print('fund coef lean,inc,money,nomoney',np.round(fund_fit(d),3))
json.dump(dict(theta=list(map(float,x)),adj=list(map(float,adj)),fund=list(map(float,fund_fit(d)))),open('data/processed/full_fit.json','w'))
d.to_csv('data/processed/hist_prepared.csv',index=False)
