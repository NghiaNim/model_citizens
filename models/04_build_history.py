"""Master race-level history 1998-2024: outcome (MIT), pres lean, FEC money/incumbency, 538+Wikipedia polls, ratings."""
import pandas as pd,numpy as np,re,os,glob
from io import StringIO
from bs4 import BeautifulSoup
import re as _re
def clean_html(h): return _re.sub(r'(rowspan|colspan)="(\d+)[^"]*"',r'\1="\2"',h)
# -------- outcomes
s=pd.read_csv('data/raw/misc/mit_senate.tsv',sep='\t')
s['stage']=s.stage.str.lower(); s['mode']=s['mode'].astype(str).str.lower(); s=s[(s.stage=='gen')&(s.writein.astype(str)!='True')&(s.candidate.astype(str).str.upper()!='OTHER')]; s=s[s['mode'].isin(['total','nan'])]
s['special']=s.special.astype(str).eq('True')
def last(n):
    n=str(n).upper(); n=re.sub(r'[",\.]|\bJR\b|\bSR\b|\bIII\b|\bII\b','',n); n=re.sub(r"\(.*?\)",'',n).strip()
    return n.split()[-1] if n else ''
s['ln']=s.candidate.map(last)
rows=[]
for (y,st,sp),g in s.groupby(['year','state_po','special']):
    g=g.groupby(['candidate','party_simplified','ln'],as_index=False).agg(v=('candidatevotes','sum'),t=('totalvotes','max')).sort_values('v',ascending=False)
    t=g.t.max(); top=g.iloc[0]
    d=g[g.party_simplified=='DEMOCRAT'].head(1); r=g[g.party_simplified=='REPUBLICAN'].head(1)
    second=g.iloc[1] if len(g)>1 else None
    rows.append(dict(year=y,st=st,special=sp,dv=d.v.sum()/t*100,rv=r.v.sum()/t*100,dname=d.ln.iloc[0] if len(d) else '',rname=r.ln.iloc[0] if len(r) else '',
        winparty=top.party_simplified,topshare=top.v/t*100,secondshare=(second.v/t*100 if second is not None else 0),secondparty=second.party_simplified if second is not None else '',topname=top.ln,secondname=second.ln if second is not None else ''))
H=pd.DataFrame(rows); H['margin']=H.dv-H.rv
H['dvr']=H.dv.gt(5)&H.rv.gt(5)&H.winparty.isin(['DEMOCRAT','REPUBLICAN'])&H.secondparty.isin(['DEMOCRAT','REPUBLICAN'])
# -------- presidential lean: state D margin minus nation; weighted 3:1 latest two
p=pd.read_csv('data/raw/misc/mit_pres.csv'); p=p[p.writein.astype(str)!='True']
g=p.groupby(['year','state_po','party_simplified']).candidatevotes.sum().unstack().fillna(0)
tot=p.groupby(['year','state_po']).totalvotes.max()
pm=((g.DEMOCRAT-g.REPUBLICAN)/tot*100).rename('m').reset_index()
nat=(g.groupby('year').sum()); natm=((nat.DEMOCRAT-nat.REPUBLICAN)/tot.groupby('year').sum()*100).rename('nat')
pm=pm.merge(natm,on='year'); pm['rel']=pm.m-pm.nat
piv=pm.pivot(index='state_po',columns='year',values='rel')
def lean(st,y):
    pr=[a for a in piv.columns if a<y][-2:]  # latest two presidential years BEFORE the election
    if y%4==0 and False: pass
    return .75*piv.loc[st,pr[-1]]+.25*piv.loc[st,pr[-2]]
H=H[H.year>=1998]; H['lean']=[lean(a,b) for a,b in zip(H.st,H.year)]
piv.to_csv('data/processed/pres_relmargin.csv'); natm.to_csv('data/processed/pres_nat.csv')
# -------- national House vote D margin for each cycle (Clerk of House / CQ)
HV={1998:-1.1,2000:0.1,2002:-4.6,2004:-2.6,2006:8.0,2008:10.6,2010:-6.8,2012:1.2,2014:-5.7,2016:-1.1,2018:8.6,2020:3.1,2022:-2.8,2024:-2.7}
H['natvote']=H.year.map(HV)
H['midterm']=H.year.mod(4)==2
PRESP={1998:1,2000:1,2002:-1,2004:-1,2006:-1,2008:-1,2010:1,2012:1,2014:1,2016:1,2018:-1,2020:-1,2022:1,2024:1}
H['president']=H.year.map(PRESP)
# -------- FEC
cols=['CAND_ID','CAND_NAME','CAND_ICI','PTY_CD','PARTY','RECEIPTS','TFA','DISB','TTA','COH_BOP','COH_COP','CAND_CONTRIB','CAND_LOANS','OLOANS','CLR','OLR','DEBT','INDIV','ST','DIST','SP','GP','GPO','RUNP','GEN','GENPCT','OTHPAC','PARTYC','END','INDREF','COMREF']
fec=[]
for f in glob.glob('data/raw/fec/weball*/weball*.txt'):
    yy=int(re.search(r'weball(\d\d)\.txt',f).group(1)); y=2000+yy
    d=pd.read_csv(f,sep='|',header=None,dtype=str,encoding='latin1',on_bad_lines='skip').iloc[:,:29]
    d.columns=cols[:29]; d=d[d.CAND_ID.str.startswith('S')]; d['year']=y; fec.append(d)
F=pd.concat(fec); F['RECEIPTS']=pd.to_numeric(F.RECEIPTS,errors='coerce').fillna(0)
F['ln']=F.CAND_NAME.map(lambda x:re.sub(r'[^A-Z\- ]','',str(x).split(',')[0].upper()).split()[-1] if str(x).split(',')[0].strip() else '')
F.to_csv('data/processed/fec_senate_all.csv',index=False)
def look(y,st,ln,which):
    if not ln: return (np.nan,'')
    g=F[(F.year==y)&(F.ST==st)&(F.ln==ln)&(F.PARTY==which)]
    if not len(g): g=F[(F.year==y)&(F.ST==st)&(F.ln==ln)]
    if not len(g): return (np.nan,'')
    g=g.sort_values('RECEIPTS',ascending=False).iloc[0]
    return (g.RECEIPTS,g.CAND_ICI)
dr=[look(a,b,c,'DEM') for a,b,c in zip(H.year,H.st,H.dname)]; rr=[look(a,b,c,'REP') for a,b,c in zip(H.year,H.st,H.rname)]
H['drcpt']=[x[0] for x in dr];H['rrcpt']=[x[0] for x in rr]
H['inc']=[1 if x[1]=='I' else (-1 if y[1]=='I' else 0) for x,y in zip(dr,rr)]
# 1998 has no FEC here; fallback from candidates' prior wins
H['logmoney']=np.log1p(H.drcpt)-np.log1p(H.rrcpt)
# -------- 538 raw polls (1998-2022)
P=pd.read_csv('data/raw/fte/raw_polls.csv'); P=P[P.type_simple=='Sen-G']
P['polldate']=pd.to_datetime(P.polldate,format='mixed'); P['electiondate']=pd.to_datetime(P.electiondate,format='mixed')
P['special']=P.race.str.contains('Sen-GS')
def dmarg(r):
    if r.cand1_party=='DEM' and r.cand2_party=='REP': return r.cand1_pct-r.cand2_pct
    if r.cand2_party=='DEM' and r.cand1_party=='REP': return r.cand2_pct-r.cand1_pct
    return np.nan
P['m']=P.apply(dmarg,axis=1)
P['und']=100-P.cand1_pct-P.cand2_pct
P['age']=P.time_to_election; P['N']=pd.to_numeric(P.samplesize,errors='coerce')
P['part']=P.partisan.fillna('')
P=P.rename(columns={'location':'st','cycle':'year','pollster':'pollster'})
keep=P[['year','st','special','pollster','polldate','age','N','part','m','und']].dropna(subset=['m'])
# independents analog for later analysis
ind=P[((P.cand1_party=='IND')&(P.cand2_party=='REP'))|((P.cand2_party=='IND')&(P.cand1_party=='REP'))].copy()
ind['im']=np.where(ind.cand1_party=='IND',ind.cand1_pct-ind.cand2_pct,ind.cand2_pct-ind.cand1_pct); ind['ia']=np.where(ind.cand1_party=='IND',ind.cand1_actual-ind.cand2_actual,ind.cand2_actual-ind.cand1_actual)
ind[ind.age<=35].groupby(['year','st']).agg(polls=('im','size'),poll=('im','mean'),act=('ia','mean')).assign(err=lambda x:x.act-x.poll).to_csv('data/processed/independent_analogs.csv'); print(pd.read_csv('data/processed/independent_analogs.csv').to_string())
# -------- 2024 polls from Wikipedia, same parser shape
names={'Arizona':'AZ','California':'CA','Connecticut':'CT','Delaware':'DE','Florida':'FL','Hawaii':'HI','Indiana':'IN','Maine':'ME','Maryland':'MD','Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT','Nebraska':'NE','Nevada':'NV','New_Jersey':'NJ','New_Mexico':'NM','New_York':'NY','North_Dakota':'ND','Ohio':'OH','Pennsylvania':'PA','Rhode_Island':'RI','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA','West_Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'}
MON={m:i+1 for i,m in enumerate("January February March April May June July August September October November December".split())}
def endd(s):
    s=str(s).replace('–','-'); ms=re.findall(r'([A-Z][a-z]+)\.?\s+(\d{1,2})(?:\s*-\s*(\d{1,2}))?',s)
    if not ms or ms[-1][0] not in MON: return pd.NaT
    m,a,b=ms[-1]; y=re.search(r'(20\d\d)',s); return pd.Timestamp(int(y.group(1)) if y else 2024,MON[m],int(b or a))
def num(x):
    m=re.search(r'(\d+(?:\.\d+)?)\s*%',str(x)); return float(m.group(1)) if m else None
r24=[]
for n,st in names.items():
    hh=H[(H.year==2024)&(H.st==st)&(~H.special)]
    if not len(hh): continue
    Dn,Rn=hh.dname.iloc[0].title(),hh.rname.iloc[0].title()
    sp=BeautifulSoup(open(f'data/raw/wiki24/{n}.html',encoding='utf8'),'lxml')
    for x in sp.select('sup.reference,.mw-editsection,span.sortkey,style'): x.decompose()
    for t in sp.select('#mw-content-text table'):
        tx=t.get_text(' ',strip=True)
        if 'Poll source' not in tx: continue
        try: d=pd.read_html(StringIO(clean_html(str(t))))[0]
        except Exception: continue
        d.columns=[' '.join(dict.fromkeys(str(y) for y in c if not str(y).startswith('Unnamed'))) if isinstance(c,tuple) else str(c) for c in d.columns]
        rc=[c for c in d.columns if Rn.lower() in c.lower()]; dc=[c for c in d.columns if Dn.lower() in c.lower()]
        if not rc or not dc: continue
        dcol=[c for c in d.columns if 'Date' in c]; sz=[c for c in d.columns if 'Sample' in c]
        srcs=[c for c in d.columns if 'Poll source' in c]
        if not dcol or not srcs: continue
        if len(rc)>1 or len(dc)>1: continue
        for _,rw in d.iterrows():
            a,b=num(rw[dc[0]]),num(rw[rc[0]])
            if a is None or b is None: continue
            e=endd(rw[dcol[0]])
            if pd.isna(e): continue
            psrc=str(rw[srcs[0]])
            nn=re.search(r'([\d,]+)\s*\(',str(rw[sz[0]])) if sz else None
            part=re.search(r'\((D|R)\)',psrc)
            r24.append(dict(year=2024,st=st,special=False,pollster=psrc,polldate=e,age=(pd.Timestamp(2024,11,5)-e).days,N=float(nn.group(1).replace(',','')) if nn else np.nan,part=part.group(1) if part else '',m=a-b,und=100-a-b))
K=pd.concat([keep,pd.DataFrame(r24)],ignore_index=True); K=K[K.age>=0]
K.to_csv('data/processed/polls_hist.csv',index=False)
print(K.groupby('year').size().to_dict())
H.to_csv('data/processed/history_base.csv',index=False)
print(H[['year','dvr']].groupby('year').sum().T.to_string())
print(H[(H.year==2024)&H.st.isin(['OH','MT','MI','NE','TX'])][['st','dname','rname','margin','lean','inc','logmoney','winparty','topname','secondname']].to_string())
print(H.logmoney.notna().mean(),H.inc.value_counts().to_dict())
