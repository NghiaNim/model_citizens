"""Clean 2026 polls: parse end-dates, sample size, population, partisan sponsor.
One row per field-period+pollster, preferring LV > RV > A and, for RCV states, the final-round head-to-head
if available (as it is what decides the winner)."""
import pandas as pd,re,numpy as np
from datetime import date
p=pd.read_csv('data/processed/polls_wiki_raw.csv')
MON={m:i+1 for i,m in enumerate("January February March April May June July August September October November December".split())}
def end(s):
    s=s.replace('–','-').replace('—','-')
    y=re.search(r'(20\d\d)',s); y=int(y.group(1)) if y else 2026
    ms=re.findall(r'([A-Z][a-z]+)\.?\s+(\d{1,2})(?:\s*-\s*(\d{1,2}))?',s)
    if not ms: return pd.NaT
    m,d1,d2=ms[-1]
    if m not in MON: return pd.NaT
    return pd.Timestamp(y,MON[m],int(d2) if d2 else int(d1))
p['end']=p.dates.map(end)
p['pop']=p.n.str.extract(r'\((LV|RV|A|V)\)')[0].fillna('U')
p['N']=pd.to_numeric(p.n.str.extract(r'([\d,]+)\s*\(')[0].str.replace(',',''),errors='coerce')
p['partisan']=p.pollster.str.extract(r'\((D|R)\)')[0].fillna('')
p['both']=p.pollster.str.contains(r'\(D\).*\(R\)|\(R\).*\(D\)',regex=True)
p.loc[p.both,'partisan']=''
p['pollster']=p.pollster.str.replace(r'\s+',' ',regex=True)
p['hkey']=p.pollster.str.replace(r'\s*\((D|R|I|L)\)','',regex=True)
p['rcv']=p.oth.str.extract(r'RCV round=([^;]+)')[0]
p['margin']=p.D-p.R
p['und']=100-p.D-p.R
ux=p.oth.astype(str).str.extract(r'Undecided[^=]*=\s*<?(\d+(?:\.\d+)?)%')[0].astype(float)
# Use the explicit undecided column when given: in multi-candidate polls 100-D-R counts third-party voters as undecided.
p['und']=ux.where(ux.notna(),p.und.clip(upper=12))
rank={'LV':0,'V':1,'RV':2,'A':3,'U':4}
p['pr']=p['pop'].map(rank)
rows=[]
for (st,hk,dates),g in p.groupby(['st','hkey','dates'],sort=False):
    g=g.sort_values('pr'); g=g[g.pr==g.pr.min()]
    if g.rcv.notna().any():
        # keep final round or BA head-to-head
        g2=g[g.rcv.astype(str).str.contains('3|BA|2',regex=True)]
        if len(g2): g=g2.sort_values('D',ascending=False).head(1) 
    else:
        # versions (e.g. with/without third parties): average them
        pass
    r=g.iloc[0].to_dict(); r['margin']=g.margin.mean(); r['und']=g.und.mean(); rows.append(r)
c=pd.DataFrame(rows); c=c[c.end.notna()]
c['age']=(pd.Timestamp(2026,11,3)-c.end).dt.days
c=c[c.end>=pd.Timestamp(2025,1,1)]
c.drop(columns=['pr','both','oth','ncol']).to_csv('data/processed/polls_clean.csv',index=False)
print(c.groupby('st').size().to_dict())
for st,g in c.groupby('st'):
    r=g[g.age<=30]
    print(st,'n30=%d'%len(r),'mean=%.1f'%r.margin.mean() if len(r) else '', 'nonpart=%.1f'%r[r.partisan==''].margin.mean() if (r.partisan=='').any() else '', 'last:',[ (x.hkey[:12],x.margin,int(x.age)) for x in g.sort_values('age').head(3).itertuples()], 'oldest age',int(g.age.max()))
