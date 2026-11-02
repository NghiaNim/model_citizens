"""Final stage: structure selection, 2026 simulation, sensitivity, decision rule, seat distribution."""
import sys;sys.path.insert(0,'models')
import numpy as np,pandas as pd,json
from scipy import stats,optimize
import core
from core import *
rng=np.random.default_rng(20261103)
def logit(p): p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def expit(x): return 1/(1+np.exp(-x))
g=lambda m:m*(1+.13*np.exp(-(m/20.)**2))
def br(p,y): return np.mean((p-y)**2)
def ll(p,y): p=np.clip(p,.003,.997); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
B=pd.read_csv('data/processed/backtest_fusion.csv'); B['mg']=g(B.mu)
def Tfn(mg,inc): return np.where((inc==1)&(mg<0)&(mg>-10),1.,np.where((inc==-1)&(mg>0)&(mg<10),-1.,0.))
B['T']=Tfn(B.mg.values,B.inc.values); TT=(B['T']!=0).values; y=(B.margin>0).values.astype(float)
x=logit(stats.norm.cdf(B.mg/B.sd))
def nl(pr,yy): pr=np.clip(pr,1e-6,1-1e-6); return -np.sum(yy*np.log(pr)+(1-yy)*np.log(1-pr))
def Pk(a,i): return expit(a[0]*x[i])
def PkT(a,i): return expit(np.where(TT[i],a[1],a[0])*x[i])
def Pd(a,i): return expit(a[0]*logit(stats.norm.cdf((B.mg.values[i]+a[1]*B['T'].values[i])/B.sd.values[i])))
VAR={'base':(Pk,[1.4],[(.5,4)]),'noisier':(PkT,[1.4,1.],[(.5,4),(.3,4)]),'shifted':(Pd,[1.4,1.],[(.5,4),(-3,10)])}
res={};full={}
for nm,(fn,x0,bd) in VAR.items():
    pr=np.zeros(len(B))
    for yr in B.year.unique():
        te=(B.year==yr).values; tr=np.where(~te)[0]
        a=optimize.minimize(lambda q:nl(fn(q,tr),y[tr]),x0,method='L-BFGS-B',bounds=bd).x; pr[te]=fn(a,np.where(te)[0])
    allv=np.arange(len(B)); full[nm]=optimize.minimize(lambda q:nl(fn(q,allv),y),x0,method='L-BFGS-B',bounds=bd).x; res[nm]=pr
    print(f"{nm:9s} LOCO LL {ll(pr,y):.4f} Brier {br(pr,y):.4f} | T-races(n={TT.sum()}) LL {ll(pr[TT],y[TT]):.3f} Brier {br(pr[TT],y[TT]):.3f} | params {np.round(full[nm],3)}")
best=min(res,key=lambda k:ll(res[k],y)); print('selected by LOCO log-loss:',best)
kk,kT=full['noisier'] if best=='noisier' else (full[best][0],full[best][0])
pll=lambda p:-(y*np.log(np.clip(p,.003,.997))+(1-y)*np.log(1-np.clip(p,.003,.997)))
dl=pd.Series(pll(res['noisier'])-pll(res['base'])).groupby(B.year.values).sum(); bs=[dl.sample(len(dl),replace=True,random_state=int(rng.integers(1e9))).sum() for _ in range(5000)]
print('noisier-base LL total %.2f 90%%CI [%.2f,%.2f]'%(dl.sum(),*np.quantile(bs,[.05,.95])))
B['pfinal']=res[best]; B['fav']=np.maximum(B.pfinal,1-B.pfinal); B['won']=((B.pfinal>.5)==(B.margin>0)).astype(int)
cal=B.groupby(pd.cut(B.fav,[.5,.6,.7,.8,.9,.95,.99,.999,1.001],right=False),observed=True).agg(n=('won','size'),pred=('fav','mean'),obs=('won','mean')).round(3);print(cal.to_string());cal.to_csv('output/calibration_backtest.csv')
print('by cycle:',{int(a):round(float(br(g_.pfinal,(g_.margin>0).astype(float))),3) for a,g_ in B.groupby('year')})
h24=B[B.year==2024];print('2024 holdout Brier %.3f LL %.3f; missed:'%(br(h24.pfinal,(h24.margin>0)*1.),ll(h24.pfinal,(h24.margin>0)*1.)),list(h24[h24.won==0].st))
B.to_csv('data/processed/backtest_final.csv',index=False)
# reference classes (leader by mg)
P=B[B.p.notna()].copy(); P['ld']=np.sign(P.mg);P['lw']=(np.sign(P.margin)==P.ld).astype(int);P['pl']=np.where(P.ld>0,P.pfinal,1-P.pfinal);P['err']=(P.margin-P.mg)*P.ld
def cb(d,B_=3000):
    ys=d.year.unique();G={a:d[d.year==a].lw.values for a in ys};o=[]
    for _ in range(B_): o.append(np.concatenate([G[a] for a in rng.choice(ys,len(ys))]).mean())
    return np.quantile(o,[.05,.95])
rows=[];c=P[(P.mg.abs()>=2)&(P.mg.abs()<8)]
cls={'All leads 2-8':c,'Leads 2-5':P[(P.mg.abs()>=2)&(P.mg.abs()<5)],'Leads 5-8':P[(P.mg.abs()>=5)&(P.mg.abs()<8)],'Leads 8-12':P[(P.mg.abs()>=8)&(P.mg.abs()<12)],
'D lead in red state (TX,OH,IA,KS,AK)':c[(c.ld>0)&(c.lean<=-8)],'D lead in a midterm (any state)':c[(c.ld>0)&(c.midterm)],'Challenger leads incumbent (ME,OH,KS,AK)':P[(P['T']!=0)&(P.mg.abs()<8)&(P.mg.abs()>=2)],'Open-seat lead (MI,TX,IA)':c[c.inc==0],'R narrow lead in red state (SC,FL,MS,LA)':P[(P.mg.abs()>=2)&(P.mg.abs()<10)&(P.ld<0)&(P.lean<=-8)]}
for nm,d in cls.items():
    ci=cb(d);rows.append(dict(cls=nm,n=len(d),cycles=d.year.nunique(),won=round(d.lw.mean(),3),lo=round(ci[0],2),hi=round(ci[1],2),model=round(d.pl.mean(),3),misses=' '.join(d[d.lw==0].year.astype(str).str[2:]+d[d.lw==0].st)))
RC=pd.DataFrame(rows);print(RC.to_string());RC.to_csv('output/reference_classes.csv',index=False)
# ---------- 2026
D=pd.read_csv('output/race_table_full.csv'); D=D.drop(columns=[c for c in D.columns if c.startswith('fav_') or c in('P_model','P_evid','P_pool','T','mgd','P_lo','P_hi','mg','P_R3','P_2020','P_D2','P_k1')],errors='ignore')
fit_=json.load(open('data/processed/full_fit.json'));th=unpack(np.array(fit_['theta']));ind=(D.kind=='I').values;ISH=-4.85
mu,v,w=mean_var(D,th);D['var']=v;D['mu_raw']=mu;D['w_poll']=w
D['mg']=g(mu);D.loc[ind,'mg']=D.mg[ind]+ISH;D['T']=Tfn(D.mg.values,D.inc.values);Td=(D['T']!=0).values
print('Trailing-incumbent 2026:',list(D[Td].st),'  k=%.3f k_T=%.3f'%(kk,kT))
def mcsim(mgd,var,S=400000,mult=1.,seed=11,sh=0.):
    r=np.random.default_rng(seed);nat=stats.t.rvs(8,size=S,random_state=r)*th['t_n']*mult/np.sqrt(8/6)+sh
    lf=r.normal(0,th['t_l'],S);dv={d:r.normal(0,th['t_r'],S) for d in np.unique(D['div'])};idio=stats.t.rvs(12,size=(S,len(D)),random_state=r)/np.sqrt(12/10);si=r.normal(0,4.5,S)
    s=np.zeros((S,len(D)))
    for i,(z,dvv,ki) in enumerate(zip(D.z,D['div'],D.kind)): s[:,i]=mgd[i]+nat+lf*z+dv[dvv]+np.sqrt(var[i])*idio[:,i]+(si if ki=='I' else 0)
    return s
def calibr(pm,a=kk,b=kT): return expit(np.where(ind,1.,np.where(Td,b,a))*logit(np.clip(pm,2.5e-6,1-2.5e-6)))
sim=mcsim(D.mg.values,D['var'].values);pm=(sim>0).mean(0);D['P_mc']=pm;D['P_model']=calibr(pm)
# band: cycle bootstrap of (k,kT)
bp=[]
for _ in range(400):
    cy=rng.choice(B.year.unique(),B.year.nunique());ii=np.concatenate([np.where(B.year==v_)[0] for v_ in cy])
    bp.append(optimize.minimize(lambda q:nl(PkT(q,ii),y[ii]),[1.4,1.],method='L-BFGS-B',bounds=[(.5,4),(.3,4)]).x)
bp=np.array(bp);print('boot k %.2f [%.2f,%.2f] k_T %.2f [%.2f,%.2f]'%(bp[:,0].mean(),*np.quantile(bp[:,0],[.05,.95]),bp[:,1].mean(),*np.quantile(bp[:,1],[.05,.95])))
band=np.array([calibr(pm,a,b) for a,b in bp]);D['P_lo'],D['P_hi']=np.quantile(band,.05,axis=0),np.quantile(band,.95,axis=0)
scen=lambda sh:calibr((mcsim(D.mg.values,D['var'].values,S=120000,sh=sh)>0).mean(0))
D['P_m3']=scen(-3.);D['P_m53']=scen(-5.3);D['P_p2']=scen(2.)
# sensitivity to poll decay
Pp=pd.read_csv('data/processed/polls_clean.csv');Pp['part']=Pp.partisan.fillna('');Pp['m']=Pp.margin;Pp['pollster']=Pp.hkey
adj=fit_['adj']
for tau in (7,28):
    core.TAU=float(tau);ag=[aggregate(Pp[Pp.st==s],adj) for s in D.st];d2=D.copy();d2['p']=[a['p'] for a in ag];d2['neff']=[a['neff'] for a in ag];d2['und']=[a['und'] for a in ag]
    m2,v2,_=mean_var(d2,th);mg2=g(m2);mg2=np.where(ind,mg2+ISH,mg2);D[f'P_tau{tau}']=calibr((mcsim(mg2,v2,S=120000)>0).mean(0));D[f'p_tau{tau}']=d2.p
core.TAU=14.
D['P_noindshift']=D.P_model
pn=(mcsim(np.where(ind,D.mg-ISH,D.mg),D['var'].values,S=120000)>0).mean(0);D['P_noindshift']=calibr(pn)
D['P_evid']=expit((logit(D.P_model)+logit(D.P_ratings))/2);D['P_pool']=expit((logit(D.P_model)+logit(D.P_ratings)+logit(D.P_market))/3)
sgn=(D.pd_user>=.5).values;F=lambda c:np.where(sgn,D[c].values,1-D[c].values)
O=pd.DataFrame(dict(race=D.st,cand=D.candidate_name,user=D.win_probability,model=F('P_model'),lo=np.where(sgn,D.P_lo,1-D.P_hi),hi=np.where(sgn,D.P_hi,1-D.P_lo),ratings=F('P_ratings'),market=F('P_market'),kalshi=F('kalshi'),poly=F('poly'),pool=F('P_pool'),polls=D.p.round(1),n30=D.n30,neff=D.neff.round(1),margin=D.mg.round(1),sd=np.sqrt(marginal_sd(D,th,D['var'].values)**2+ind*4.5**2).round(1),rate=D.rate_score.round(2),inc=D.inc,kind=D.kind,trailing=Td,R3=F('P_m3'),R53=F('P_m53'),D2=F('P_p2'),tau7=F('P_tau7'),tau28=F('P_tau28'),noind=F('P_noindshift'),lean=D.lean.round(1),money=D.money.round(2),drcpt=D.drcpt,rrcpt=D.rrcpt))
# DECISION RULE (written after preliminary runs, so NOT a true pre-registration; frozen for the final run): change a non-100 number only if ALL hold:
#  (1) |pool-user| >= 4 pts; (2) user lies outside the model's 90% band; (3) ratings and model on same side of the user;
#  (4) a historical reference class with n>=20 backs the race: i.e. not a trailing-incumbent race and not an independent-candidate race.
#  (5) sign of the pick unchanged.  New value = pool rounded. For a 100: keep if pool>=.995, otherwise pool rounded (a 100 claims >=99.5%).
def dec(r):
    if r.user==100:
        return (100,'keep (≥99.5%)') if r.pool>=.995 else (int(round(r.pool*100)),'trim: below 99.5%')
    c1=abs(r.pool*100-r.user)>=4;c2=(r.user/100<r.lo) or (r.user/100>r.hi);c3=(r.model-r.user/100)*(r.ratings-r.user/100)>0;c4=(not r.trailing) and r.kind!='I';c5=r.pool>.5
    if c1 and c2 and c3 and c4 and c5:return (int(round(r.pool*100)),'update')
    return (int(r.user),'keep'+('' if c1 else ' (within 4)')+(' trailing-inc' if r.trailing else '')+(' independent' if r.kind=='I' else '')+('' if c3 else ' sources split'))
de=O.apply(dec,axis=1);O['proposed']=[a for a,b in de];O['decision']=[b for a,b in de]
pd.set_option('display.width',280)
print(O[['race','cand','user','model','lo','hi','ratings','market','pool','proposed','decision','polls','n30','margin','sd','R53','tau7','tau28','noind']].round(3).sort_values(['user','race']).to_string())
O.to_csv('output/final_comparison.csv',index=False);D.to_csv('output/race_table_full.csv',index=False)
print('agree user/market non100: mean abs',np.round((D[D.win_probability<100].pd_user-D[D.win_probability<100].P_market).abs().mean()*100,2),'corr logit',np.round(np.corrcoef(logit(D[D.win_probability<100].pd_user),logit(D[D.win_probability<100].P_market))[0,1],4))
# joint
dside=(D.kind=='D').values
def joint(marg,mult=1.,S=400000):
    s=mcsim(D.mg.values,D['var'].values,S=S,mult=mult,seed=21)
    for i in range(len(D)): s[:,i]-=np.quantile(s[:,i],1-np.clip(marg[i],1e-4,1-1e-4))
    w_=s>0;d=34+w_[:,dside].sum(1);i_=w_[:,~dside].sum(1);return d,i_
pdprop=np.where(sgn,O.proposed/100,1-O.proposed/100)
OUT={}
for nm,m in (('model',D.P_model.values),('pool',D.P_pool.values),('market',np.nan_to_num(D.P_market.values,nan=.86)),('user',D.pd_user.values),('proposed',pdprop)):
    d,i_=joint(m);OUT[nm]=dict(mean=float(d.mean()),med=int(np.median(d)),q10=float(np.quantile(d,.1)),q90=float(np.quantile(d,.9)),p51=float((d>=51).mean()),p50=float((d==50).mean()))
    print(nm,{k:round(v_,3) for k,v_ in OUT[nm].items()},'P(indep wins≥1)=%.3f'%(i_>0).mean())
    if nm=='proposed': pd.Series(d).value_counts(normalize=True).sort_index().rename('p').to_csv('output/seat_distribution.csv')
for mu_ in (1.,1.5,2.,3.):
    d,_=joint(np.nan_to_num(D.P_market.values,nan=.86),mult=mu_,S=120000);print(' market marginals nat sd x%.1f: P(D>=51)=%.3f'%(mu_,(d>=51).mean()))
json.dump(OUT,open('output/seat_summary.json','w'),indent=1)
print('Competitive-race correlation (proposed marginals):');s=mcsim(D.mg.values,D['var'].values,S=100000,seed=9)
print(pd.DataFrame(s>0,columns=D.st)[['TX','OH','IA','KS','ME','MI','AK']].corr().round(2).to_string())
# expected scores
Up=O.user/100;Pr=O.proposed/100
def eb(f,t):return np.sum(t*(1-f)**2+(1-t)*f**2)
def el(f,t):f=np.clip(f,.005,.995);return -np.sum(t*np.log(f)+(1-t)*np.log(1-f))
rows=[]
for nm,t in (('model',O.model),('ratings',O.ratings),('market',O.market),('pool',O.pool)):
    rows.append(dict(truth=nm,brier_user=eb(Up,t),brier_prop=eb(Pr,t),logl_user=el(Up,t),logl_prop=el(Pr,t)))
ES=pd.DataFrame(rows).round(3);print(ES.to_string());ES.to_csv('output/expected_score.csv',index=False)
