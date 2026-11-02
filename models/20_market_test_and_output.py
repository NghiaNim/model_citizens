"""(a) Test Polymarket as an input on 2024 Senate races. (b) Write the pipeline's own predictions: three-source log-odds average for every race."""
import numpy as np,pandas as pd,json
from scipy import stats
def logit(p):p=np.clip(p,1e-4,1-1e-4);return np.log(p/(1-p))
def expit(x):return 1/(1+np.exp(-x))
B=pd.read_csv('data/processed/backtest_final.csv');B=B[B.year==2024].copy();B['pm']=expit(1.051*np.log(np.clip(stats.norm.cdf(B.mu/B.sd),1e-6,1-1e-6)/(1-np.clip(stats.norm.cdf(B.mu/B.sd),1e-6,1-1e-6))))
Rt=pd.read_csv('data/processed/ratings_hist.csv');Rt=Rt[Rt.year==2024];B=B.merge(Rt[['st','score']],on='st',how='left');B['pr']=expit(1.642*B.score)
names=dict(AZ='Arizona',CA='California',CT='Connecticut',DE='Delaware',FL='Florida',HI='Hawaii',IN='Indiana',ME='Maine',MD='Maryland',MA='Massachusetts',MI='Michigan',MN='Minnesota',MS='Mississippi',MO='Missouri',MT='Montana',NE='Nebraska',NV='Nevada',NJ='New Jersey',NM='New Mexico',NY='New York',ND='North Dakota',OH='Ohio',PA='Pennsylvania',RI='Rhode Island',TN='Tennessee',TX='Texas',UT='Utah',VT='Vermont',VA='Virginia',WA='Washington',WV='West Virginia',WI='Wisconsin',WY='Wyoming')
M=pd.DataFrame(json.load(open('data/raw/markets/polymarket_2024_history.json')));rows=[]
for r in B.itertuples():
    g=M[(M.st==r.st)&M.title.str.startswith(names.get(r.st,'zz'))]
    if not len(g):continue
    q=g.q.astype(str).str.lower()
    d=g[q.str.contains('democrat')|q.str.contains(str(r.dname).lower())];x=g[q.str.contains('republican')|q.str.contains(str(r.rname).lower())]
    if len(d) and len(x):rows.append(dict(st=r.st,pk=float(d.p.iloc[0])/(float(d.p.iloc[0])+float(x.p.iloc[0])),vol=d.vol.iloc[0]))
B=B.merge(pd.DataFrame(rows).drop_duplicates('st'),on='st');y=(B.margin>0)*1.
def sc(p,y):return np.mean((p-y)**2),-np.mean(y*np.log(np.clip(p,.003,.997))+(1-y)*np.log(1-np.clip(p,.003,.997)))
B['pool']=expit((logit(B.pm)+logit(B.pr.fillna(.5))+logit(B.pk))/3);B['two']=expit((logit(B.pm)+logit(B.pr.fillna(.5)))/2)
print('2024 races with all sources n=%d'%len(B));
for c in ('pm','pr','pk','two','pool'):print(c,'Brier %.4f LL %.4f wins called %d/%d'%(*sc(B[c],y),((B[c]>.5)==(B.margin>0)).sum(),len(B)))
s=B[(B.pk>.05)&(B.pk<.95)];ys=(s.margin>0)*1.;print('competitive n=%d'%len(s),{c:(round(sc(s[c],ys)[0],3)) for c in ('pm','pr','pk','two','pool')});print(s[['st','margin','pm','pr','pk','pool']].round(3).to_string())
hi=B[np.maximum(B.pk,1-B.pk)>=.9];print('market>=90%% n=%d favourite won %d'%(len(hi),(np.sign(hi.margin)==np.sign(hi.pk-.5)).sum()))
B[['st','margin','pm','pr','pk','pool']].to_csv('output/market_test_2024.csv',index=False)
# (b)
O=pd.read_csv('output/final_comparison.csv');D=pd.read_csv('output/race_table_full.csv');U=pd.read_csv('output/predictions_original_backup.csv');U['race']=U.race_id.str[:2]
sgn=(D.pd_user>=.5).values
pd_=np.where(sgn,O.pool,1-O.pool) # P(D-side candidate wins)
R=U.merge(pd.DataFrame(dict(race=O.race,pd=pd_,kalshi=D.kalshi,poly=D.poly)),on='race')
def out(r):
    p=r.pd;dside=p>=.5;party=r.predicted_winner_party;nm=r.candidate_name
    userD=party!='Republican'
    if dside!=userD:
        nm={'IA':'Ashley Hinson'}[r.race];party='Republican'
    return pd.Series(dict(race_id=r.race_id,chamber=r.chamber,predicted_winner_party=party,candidate_name=nm,win_probability=int(np.floor(max(p,1-p)*100+.5))))
Fin=R.apply(out,axis=1);Fin.to_csv('output/predictions_pipeline.csv',index=False)
T=Fin.copy();T['race']=T.race_id.str[:2];T=T.merge(O[['race','polls','n30','model','ratings','market','kalshi','poly','lean']],on='race').merge(pd.DataFrame(dict(race=O.race,flip=(np.where(pd_>=.5,1,0)!=np.where(sgn,1,0)))),on='race')
for c in ('model','ratings','market','kalshi','poly'):T[c]=np.where(T.flip,1-T[c],T[c])
T=T.sort_values('win_probability')
for r in T.itertuples():print(f"| {r.race} | {r.candidate_name} ({r.predicted_winner_party[0]}) | {r.win_probability} | {r.model*100:.1f} | {r.ratings*100:.1f} | {'—' if pd.isna(r.kalshi) else f'{r.kalshi*100:.1f}'} | {r.poly*100:.1f} | {'—' if pd.isna(r.polls) else f'{r.polls:+.1f}'} | {r.lean:+.0f} |")
print(Fin.predicted_winner_party.value_counts().to_dict(),(Fin.win_probability==100).sum())
print(json.load(open('output/seat_summary.json'))['pool'])
