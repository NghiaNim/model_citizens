"""Revised numbers after governor replication. Same rule, minus the parameter band (not re-bootstrapped).
Unpolled safe races keep earlier values: governor races say nothing about races without polls."""
import pandas as pd,numpy as np
T=pd.read_csv('output/revised_after_replication.csv');O=pd.read_csv('output/final_comparison.csv');U=pd.read_csv('output/predictions_original_backup.csv');U['race']=U.race_id.str[:2]
m=O.merge(T[['race','combined k','combined k model']],on='race').rename(columns={'combined k':'pool2','combined k model':'model2'})
def dec(r):
    polled=not pd.isna(r.polls)
    if r.user==100:
        if not polled:return r.proposed,'earlier value: no polls'
        return (100,'keep') if r.pool2>=.995 else (int(round(r.pool2*100)),'trim')
    c1=abs(r.pool2*100-r.user)>=4;c3=(r.model2-r.user/100)*(r.ratings-r.user/100)>0;c4=(not r.trailing) and r.kind!='I';c5=r.pool2>.5
    return (int(round(r.pool2*100)),'update') if (c1 and c3 and c4 and c5) else (int(r.user),'keep')
d=m.apply(dec,axis=1);m['revised']=[a for a,b in d];m['why']=[b for a,b in d]
print(m[m.user<100][['race','user','proposed','revised','why','model2','ratings','market','pool2']].round(3).sort_values('user').to_string())
print(m[(m.user==100)&(m.revised!=100)][['race','proposed','revised','model2','pool2','polls']].round(3).sort_values('revised').to_string())
print('100s kept',(m.revised==100).sum(),list(m[m.revised==100].race))
out=U.merge(m[['race','revised']],on='race');out['win_probability']=out.revised;out[['race_id','chamber','predicted_winner_party','candidate_name','win_probability']].to_csv('output/predictions_revised.csv',index=False)
m.to_csv('output/revised_decisions.csv',index=False)
