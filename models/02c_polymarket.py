import json,urllib.request,urllib.parse,time
H={'User-Agent':'Mozilla/5.0'}
names=dict(AK='Alaska',AL='Alabama',AR='Arkansas',CO='Colorado',DE='Delaware',FL='Florida',GA='Georgia',IA='Iowa',ID='Idaho',IL='Illinois',KS='Kansas',KY='Kentucky',LA='Louisiana',MA='Massachusetts',ME='Maine',MI='Michigan',MN='Minnesota',MS='Mississippi',MT='Montana',NC='North Carolina',NE='Nebraska',NH='New Hampshire',NJ='New Jersey',NM='New Mexico',OH='Ohio',OK='Oklahoma',OR='Oregon',RI='Rhode Island',SC='South Carolina',SD='South Dakota',TN='Tennessee',TX='Texas',VA='Virginia',WV='West Virginia',WY='Wyoming')
out={}
for st,n in names.items():
    u='https://gamma-api.polymarket.com/public-search?'+urllib.parse.urlencode(dict(q=f'{n} Senate election winner 2026',limit_per_type=10,events_status='active'))
    try: d=json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=30))
    except Exception as e: print(st,'ERR',e); continue
    rows=[]
    for e in d.get('events',[]):
        t=e.get('title','')
        if n.lower() in t.lower() and 'senate' in t.lower() and 'primary' not in t.lower() and not any(w in t.lower() for w in ('margin','vote','by how','turnout','win by','runoff','control','nominee')):
            for m in e.get('markets',[]):
                try:
                    pr=json.loads(m['outcomePrices']); rows.append(dict(ev=t,q=m.get('groupItemTitle') or m.get('question'),yes=float(pr[0]),bid=m.get('bestBid'),ask=m.get('bestAsk'),vol=m.get('volumeNum'),closed=m.get('closed')))
                except Exception: pass
    out[st]=rows; time.sleep(.15)
json.dump(out,open('data/raw/markets/polymarket_snapshot.json','w'),indent=0)
for st,r in out.items():
    print(st,[(x['q'][:22],round(x['yes'],3),int(x['vol'] or 0)) for x in r if x['yes']>0.005 and not x['closed']][:5], (r[0]['ev'][:45] if r else 'NONE'))
