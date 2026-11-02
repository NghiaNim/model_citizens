"""Pull pre-election 2024 Senate prices from Polymarket's public price history, to test market prices as an input."""
import json,urllib.request,urllib.parse,time,datetime as dt
H={'User-Agent':'Mozilla/5.0'}
def get(u):return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=30))
names=dict(AZ='Arizona',CA='California',CT='Connecticut',DE='Delaware',FL='Florida',HI='Hawaii',IN='Indiana',ME='Maine',MD='Maryland',MA='Massachusetts',MI='Michigan',MN='Minnesota',MS='Mississippi',MO='Missouri',MT='Montana',NE='Nebraska',NV='Nevada',NJ='New Jersey',NM='New Mexico',NY='New York',ND='North Dakota',OH='Ohio',PA='Pennsylvania',RI='Rhode Island',TN='Tennessee',TX='Texas',UT='Utah',VT='Vermont',VA='Virginia',WA='Washington',WV='West Virginia',WI='Wisconsin',WY='Wyoming')
ts_lo=int(dt.datetime(2024,11,2,tzinfo=dt.timezone.utc).timestamp());ts_hi=int(dt.datetime(2024,11,5,12,tzinfo=dt.timezone.utc).timestamp())
out=[]
for st,n in names.items():
    found=False
    for q in (f'{n} Senate Election Winner',f'Will a Democrat win {n} Senate 2024',f'{n} Senate 2024'):
        try:d=get('https://gamma-api.polymarket.com/public-search?'+urllib.parse.urlencode(dict(q=q,limit_per_type=15,events_status='closed')))
        except Exception as e:continue
        for e in d.get('events',[]):
            t=e.get('title','')
            if n.lower() in t.lower() and 'senate' in t.lower() and '2024' in (e.get('endDate','')+t) and 'primary' not in t.lower():
                for m in e.get('markets',[]):
                    ids=m.get('clobTokenIds')
                    if not ids:continue
                    tok=json.loads(ids)[0]
                    try:h=get(f'https://clob.polymarket.com/prices-history?market={tok}&startTs={ts_lo}&endTs={ts_hi}&fidelity=60').get('history',[])
                    except Exception:h=[]
                    if h:
                        out.append(dict(st=st,title=t,q=m.get('groupItemTitle') or m.get('question'),p=h[-1]['p'],t=h[-1]['t'],vol=m.get('volumeNum')));found=True
                    time.sleep(.1)
        if found:break
    print(st,found,flush=True)
json.dump(out,open('data/raw/markets/polymarket_2024_history.json','w'),indent=0)
for r in out:print(r['st'],r['title'][:38],'|',str(r['q'])[:30],round(r['p'],3),int(r['vol'] or 0))
