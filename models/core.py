"""Shared model code: poll aggregation, fundamentals, covariance likelihood, predictive distribution."""
import numpy as np,pandas as pd
from scipy.optimize import minimize
from scipy import stats
DIV={'CT':1,'ME':1,'MA':1,'NH':1,'RI':1,'VT':1,'NJ':2,'NY':2,'PA':2,'IL':3,'IN':3,'MI':3,'OH':3,'WI':3,'IA':4,'KS':4,'MN':4,'MO':4,'NE':4,'ND':4,'SD':4,'DE':5,'FL':5,'GA':5,'MD':5,'NC':5,'SC':5,'VA':5,'WV':5,'AL':6,'KY':6,'MS':6,'TN':6,'AR':7,'LA':7,'OK':7,'TX':7,'AZ':8,'CO':8,'ID':8,'MT':8,'NV':8,'NM':8,'UT':8,'WY':8,'AK':9,'CA':9,'HI':9,'OR':9,'WA':9,'DC':5}
TAU=14.; WINDOW=60
def aggregate(pl,partisan_adj=(0.,0.)):
    """pl: polls for one race with cols age,N,part,m,und,pollster. Exponential decay, sqrt-N, partisan discount, pollster-repeat discount."""
    pl=pl[(pl.age<=WINDOW)&pl.m.notna()].copy()
    if not len(pl): return dict(p=np.nan,neff=0.,und=np.nan,npolls=0)
    N=pl.N.fillna(600).clip(100,3000)
    w=np.exp(-pl.age/TAU)*np.sqrt(N/600.)*np.where(pl.part.fillna('')!='',0.5,1.)
    pc=pl.pollster.astype(str).str.replace(r'\s*\((D|R)\)','',regex=True)
    w=w/np.sqrt(pc.map(pc.value_counts()))
    adj=np.where(pl.part=='D',-partisan_adj[0],np.where(pl.part=='R',-partisan_adj[1],0.))
    p=np.sum(w*(pl.m+adj))/np.sum(w); und=np.sum(w*pl.und.clip(0,40))/np.sum(w)
    neff=np.sum(w)**2/np.sum(w**2)
    return dict(p=p,neff=neff*min(1.,np.sum(w)/0.7),und=und,npolls=len(pl))   # stale-only samples get less credit
def estimate_partisan_adj(P):
    P=P[P.age<=WINDOW].copy(); P['rm']=P.groupby(['year','st','special']).m.transform('median')
    P['d']=P.m-P.rm; return (P[P.part=='D'].d.mean(),P[P.part=='R'].d.mean())
def fund_fit(D):
    """within-cycle OLS: y ~ cycle FE + b1 lean + b2 inc + b3 money + b4 money_missing"""
    X=D[['lean','inc','money','nomoney']].values.astype(float); y=D.margin.values.astype(float); c=D.year.values
    Xd=X-pd.DataFrame(X).groupby(c).transform('mean').values; yd=y-pd.Series(y).groupby(c).transform('mean').values
    b,*_=np.linalg.lstsq(Xd,yd,rcond=None); return b
def fund_raw(D,b): return D[['lean','inc','money','nomoney']].values.astype(float)@b
def env_estimate(p,f0,neff,prior_mu=0.,prior_sd=8.,obs_sd=7.):
    """Precision-weighted national environment from races with polls; shrunk to prior."""
    ok=~np.isnan(p)
    if ok.sum()==0: return prior_mu,prior_sd
    prec=(neff[ok]/(neff[ok]+1.))/obs_sd**2
    post_prec=prec.sum()+1/prior_sd**2
    return (prior_mu/prior_sd**2+np.sum(prec*(p[ok]-f0[ok])))/post_prec, post_prec**-.5
NP=9
def unpack(t):
    return dict(s_p=np.exp(t[0]),s_n=np.exp(t[1]),s_f=np.exp(t[2]),t_n=np.exp(t[3]),t_l=np.exp(t[4]),t_r=np.exp(t[5]),lam=t[6],lam2=t[7],s_u=np.exp(t[8]),c_p=np.exp(t[9]),c_f=np.exp(t[10]))
INIT=np.r_[np.log([3.5,3.,6.,2.5,1.5,1.5]),0.,0.,np.log(.1),np.log(.15),np.log(.3)]
def adj_poll(df,th):
    return df.p.values+th['lam']*df.und.fillna(0).values*df.z.values+th['lam2']*df.und.fillna(0).values*df.oppwind.values
def mean_var(df,th,polls_only=False):
    p=adj_poll(df,th); f=df.f.values; has=~np.isnan(p)
    vp=np.where(has,th['s_p']**2+th['s_n']**2/np.maximum(df.neff.values,.2)+(th['s_u']*df.und.fillna(0).values)**2+(th['c_p']*np.nan_to_num(p))**2,np.inf)
    vf=(th['s_f']**2+(th['c_f']*f)**2)*df.fscale.values**2
    if polls_only: vf=vf*1e6
    w=np.where(has,(1/vp)/(1/vp+1/vf),0.)
    mu=np.where(has,w*np.nan_to_num(p)+(1-w)*f,f)
    v=np.where(has,1/(1/vp+1/vf),vf)
    return mu,v,w
def cov_cycle(df,th,v):
    z=df.z.values; dv=df['div'].values
    K=th['t_n']**2+th['t_l']**2*np.outer(z,z)+th['t_r']**2*(dv[:,None]==dv[None,:])+np.diag(v)
    return K
def nll(t,df,polls_only=False):
    th=unpack(t); tot=0.
    mu,v,_=mean_var(df,th,polls_only)
    r=df.margin.values-mu
    for y,idx in df.groupby('year').indices.items():
        K=cov_cycle(df.iloc[idx],th,v[idx])
        L=np.linalg.cholesky(K+1e-8*np.eye(len(idx)))
        a=np.linalg.solve(L,r[idx])
        tot+=.5*a@a+np.log(np.diag(L)).sum()
    # weak priors: keep scales finite & lam small
    tot+=.5*((t[6]/.5)**2+(t[7]/.5)**2)
    return tot
def fit(df,x0=None,polls_only=False):
    x0=INIT if x0 is None else x0
    res=minimize(nll,x0,args=(df,polls_only),method='L-BFGS-B',bounds=[(-3,3.5)]*6+[(-1.5,1.5)]*2+[(-5,1)]+[(-6,0.5)]*2,options=dict(maxiter=400))
    return res.x,res
def marginal_sd(df,th,v):
    z=df.z.values
    return np.sqrt(th['t_n']**2+th['t_l']**2*z**2+th['t_r']**2+v)
