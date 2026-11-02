import json,urllib.request,time,re,csv
st="AK AL AR CO DE FL GA IA ID IL KS KY LA MA ME MI MN MS MT NC NE NH NJ NM OH OK OR RI SC SD TN TX VA WV WY".split()
H={'User-Agent':'Mozilla/5.0'}
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u,headers=H),timeout=30))
rows=[]
for s in st:
    for ev in [f'SENATE{s}-26',f'SENATE{s}S-26',f'SENATE{s}SPECIAL-26']:
        try: d=get(f'https://api.elections.kalshi.com/trade-api/v2/markets?event_ticker={ev}&limit=50')
        except Exception as e: continue
        for m in d.get('markets',[]):
            rows.append(dict(st=s,event=ev,ticker=m['ticker'],title=m.get('yes_sub_title') or m.get('subtitle'),bid=m.get('yes_bid_dollars',m.get('yes_bid')),ask=m.get('yes_ask_dollars',m.get('yes_ask')),last=m.get('last_price_dollars',m.get('last_price')),vol=m.get('volume_fp',m.get('volume')),oi=m.get('open_interest_fp',m.get('open_interest')),status=m.get('status')))
        time.sleep(.1)
json.dump(rows,open('data/raw/markets/kalshi_snapshot.json','w'),indent=0)
for r in rows: print(r['st'],r['title'],r['bid'],r['ask'],r['last'],r['vol'])
