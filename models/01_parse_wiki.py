"""Parse each Wikipedia race page -> polls, ratings, fundraising, past results.
Only tables inside the 'General election' H2 section are used; a poll table qualifies only
if BOTH of the actual nominees are columns (hypothetical match-ups are excluded this way)."""
import re,json
from io import StringIO
from bs4 import BeautifulSoup
import re as _re
def clean_html(h): return _re.sub(r'(rowspan|colspan)="(\d+)[^"]*"',r'\1="\2"',h)
import pandas as pd
NOM={'AK':('Sullivan','Peltola'),'AL':('Moore','Wess'),'AR':('Cotton','Shoffner'),'CO':('Baisley','Hickenlooper'),'DE':('Katz','Coons'),
'FL':('Moody','Nixon'),'GA':('Collins','Ossoff'),'IA':('Hinson','Turek'),'ID':('Risch','Achilles'),'IL':('Tracy','Stratton'),
'KS':('Marshall','Hamilton'),'KY':('Barr','Booker'),'LA':('Letlow','Davis'),'MA':('Deaton','Markey'),'ME':('Collins','Jackson'),
'MI':('Rogers','El-Sayed'),'MN':('Tafoya','Flanagan'),'MS':('Hyde-Smith','Colom'),'MT':('Alme','Bodnar'),'NC':('Whatley','Cooper'),
'NE':('Ricketts','Osborn'),'NH':('Sununu','Pappas'),'NJ':('Murphy','Booker'),'NM':('Marker','Luj'),'OH':('Husted','Brown'),
'OK':('Hern','Thomas'),'OR':('Smith','Merkley'),'RI':('McKay','Reed'),'SC':('Graham','Andrews'),'SD':('Rounds','Bengs'),
'TN':('Hagerty','Bradshaw'),'TX':('Paxton','Talarico'),'VA':('Mizusawa','Warner'),'WV':('Capito','Anderson'),'WY':('Hageman','Byrd')}
def flat(c):
    if isinstance(c,tuple):
        c=[str(x) for x in c if not str(x).startswith('Unnamed')]
        return ' '.join(dict.fromkeys(c))
    return str(c)
def num(x):
    m=re.search(r'(\d+(?:\.\d+)?)\s*%',str(x)); return float(m.group(1)) if m else None
polls=[];ratings=[];funds=[]
for st,(R,D) in NOM.items():
    s=BeautifulSoup(open(f'data/raw/wiki/{st}.html',encoding='utf8'),'lxml')
    for sup in s.select('sup.reference, .mw-editsection, span.sortkey, style'): sup.decompose()
    body=s.select_one('#mw-content-text')
    # walk to General election section
    h=[x for x in body.select('h2') if 'General election' in x.get_text()]
    if not h: print('no GE',st); continue
    sec=[];node=h[0].parent if h[0].parent.name=='div' else h[0]
    for sib in node.next_siblings:
        if getattr(sib,'name',None)=='h2' or (getattr(sib,'name',None)=='div' and sib.find('h2',recursive=False)): break
        sec.append(sib)
    for el in sec:
        if not getattr(el,'find_all',None): continue
        tabs=[el] if el.name=='table' else el.find_all('table')
        for t in tabs:
            txt=t.get_text(' ',strip=True)
            try: df=pd.read_html(StringIO(clean_html(str(t))),flavor='lxml')[0]
            except Exception: continue
            df.columns=[flat(c) for c in df.columns]
            cols=[c for c in df.columns]
            cl=' | '.join(cols)
            if 'Source' in cl and ('anking' in cl or 'Rating' in cl) :
                for _,r in df.iterrows(): ratings.append(dict(st=st,source=str(r.iloc[0]),rating=str(r.iloc[1]),asof=str(r.iloc[2]) if len(r)>2 else ''))
            if 'Poll source' in cl and any(R in c for c in cols) and any(D in c for c in cols):
                rc=[c for c in cols if R in c][0]; dc=[c for c in cols if D in c][0]
                dcol=[c for c in cols if 'Date' in c][0]; 
                sz=[c for c in cols if 'Sample' in c]; moe=[c for c in cols if 'argin' in c]
                others=[c for c in cols if c not in (rc,dc,dcol,'Poll source') and c not in sz and c not in moe]
                for _,r in df.iterrows():
                    if num(r[rc]) is None or num(r[dc]) is None: continue
                    polls.append(dict(st=st,pollster=str(r['Poll source']),dates=str(r[dcol]),n=str(r[sz[0]]) if sz else '',moe=str(r[moe[0]]) if moe else '',
                                      R=num(r[rc]),D=num(r[dc]),oth='; '.join(f'{c}={r[c]}' for c in others)[:160],ncol=len(cols)))
            if 'Raised' in cl or 'raised' in cl:
                funds.append(dict(st=st,table=df.to_dict('records')))
pd.DataFrame(polls).to_csv('data/processed/polls_wiki_raw.csv',index=False)
pd.DataFrame(ratings).to_csv('data/processed/ratings_wiki_raw.csv',index=False)
json.dump(funds,open('data/processed/funds_wiki_raw.json','w'),default=str,indent=0)
p=pd.DataFrame(polls)
print(p.groupby('st').size().to_dict())
print(pd.DataFrame(ratings).groupby('st').size().to_dict())
print(len(funds),[f['st'] for f in funds])
