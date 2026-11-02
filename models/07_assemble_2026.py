import sys;sys.path.insert(0,'models')
import numpy as np,pandas as pd,json,re
from core import *
import statsmodels.api as sm
R=pd.read_csv('data/processed/backtest_fusion.csv')
for lim in (10,15,25,200):
    g=R[R.p.notna()&(R.mu.abs()<lim)]
    m=sm.OLS(g.margin,sm.add_constant(g[['mu']])).fit(cov_type='cluster',cov_kwds={'groups':g.year})
    print(f"margin ~ mu | |mu|<{lim} n={len(g)} beta={m.params['mu']:.3f} (se {m.bse['mu']:.3f}) const={m.params['const']:.2f} (se {m.bse['const']:.2f}) resid sd={np.sqrt(m.scale):.2f}")
fit_=json.load(open('data/processed/full_fit.json')); adj=fit_['adj'];b=np.array(fit_['fund']);th=unpack(np.array(fit_['theta']))
NOM={'AK':('Sullivan','Peltola','D',-1),'AL':('Moore','Wess','D',0),'AR':('Cotton','Shoffner','D',-1),'CO':('Baisley','Hickenlooper','D',1),'DE':('Katz','Coons','D',1),
'FL':('Moody','Nixon','D',-1),'GA':('Collins','Ossoff','D',1),'IA':('Hinson','Turek','D',0),'ID':('Risch','Achilles','I',-1),'IL':('Tracy','Stratton','D',0),
'KS':('Marshall','Hamilton','D',-1),'KY':('Barr','Booker','D',0),'LA':('Letlow','Davis','D',0),'MA':('Deaton','Markey','D',1),'ME':('Collins','Jackson','D',-1),
'MI':('Rogers','El-Sayed','D',0),'MN':('Tafoya','Flanagan','D',0),'MS':('Hyde-Smith','Colom','D',-1),'MT':('Alme','Bodnar','I',0),'NC':('Whatley','Cooper','D',0),
'NE':('Ricketts','Osborn','I',-1),'NH':('Sununu','Pappas','D',0),'NJ':('Murphy','Booker','D',1),'NM':('Marker','Lujan','D',1),'OH':('Husted','Brown','D',-1),
'OK':('Hern','Thomas','D',0),'OR':('Smith','Merkley','D',1),'RI':('McKay','Reed','D',1),'SC':('Graham','Andrews','D',-1),'SD':('Rounds','Bengs','I',-1),
'TN':('Hagerty','Bradshaw','D',-1),'TX':('Paxton','Talarico','D',0),'VA':('Mizusawa','Warner','D',1),'WV':('Capito','Anderson','D',-1),'WY':('Hageman','Byrd','D',0)}
piv=pd.read_csv('data/processed/pres_relmargin.csv',index_col=0); piv.columns=piv.columns.astype(int)
F=pd.read_csv('data/processed/fec_senate_all.csv',low_memory=False); F=F[F.year==2026]
print('FEC 2026 latest coverage end date:',pd.to_datetime(F.END,errors='coerce',format='%m/%d/%Y').max())
rows=[]
P=pd.read_csv('data/processed/polls_clean.csv'); P['part']=P.partisan.fillna(''); P['m']=P.margin; P['pollster']=P.hkey
for st,(r,d,kind,inc) in NOM.items():
    def rc(n,party):
        n=n.upper(); g=F[(F.ST==st)&(F.CAND_NAME.str.upper().str.contains(n.split('-')[-1] if n=='EL-SAYED' else n,regex=False,na=False))]
        pg=g[g.PARTY==party] if len(g[g.PARTY==party]) else g
        return float(pg.RECEIPTS.max()) if len(pg) else np.nan
    dr=rc(d,'DEM' if kind=='D' else 'IND'); rr=rc(r,'REP')
    pl=P[P.st==st]; a=aggregate(pl,adj)
    a30=pl[pl.age<=30]
    rows.append(dict(st=st,rname=r,dname=d,kind=kind,inc=inc,lean=.75*piv.loc[st,2024]+.25*piv.loc[st,2020],drcpt=dr,rrcpt=rr,**a,
      raw30=a30.m.mean(),n30=len(a30),nonpart30=a30[a30.part==''].m.mean(),lastpoll=pl.end.max() if len(pl) else ''))
D=pd.DataFrame(rows)
D['year']=2026;D['special']=D.st.isin(['FL','OH']);D['money']=(np.log1p(D.drcpt)-np.log1p(D.rrcpt)).clip(-3,3);D['nomoney']=D.money.isna().astype(float);D['money']=D.money.fillna(0)
D['z']=D.lean/15.;D['div']=D.st.map(DIV);D['oppwind']=1.;D['fscale']=np.where(D.kind=='I',1.5,1.);D['neff']=D.neff.fillna(0)
D['f0']=fund_raw(D,b); print('lean/money'); 
print(D[['st','kind','inc','lean','drcpt','rrcpt','money','f0','p','neff','und','n30','raw30','nonpart30']].round(1).to_string())
D.to_csv('data/processed/race_2026_base.csv',index=False)
e,es=env_estimate(D[D.kind=='D'].p.values,D[D.kind=='D'].f0.values,D[D.kind=='D'].neff.values)
print('poll-implied national environment e=%.2f sd=%.2f (fund intercept on within-cycle scale)'%(e,es))
Hp=pd.read_csv('data/processed/hist_prepared.csv')
hv=Hp.groupby('year').apply(lambda g:pd.Series(dict(env=(g.f-g.f0).mean(),nat=g.natvote.mean())),include_groups=False); print(hv.round(1).T.to_string())
fit2=np.polyfit(hv.nat,hv.env,1); print('env~natvote slope/int',np.round(fit2,2),' corr',round(np.corrcoef(hv.nat,hv.env)[0,1],2),' GB=+8 implies env',round(np.polyval(fit2,8),1))
