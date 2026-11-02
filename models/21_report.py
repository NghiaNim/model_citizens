"""Builds output/report.html, output/justification_by_race.csv and output/seat_distribution.csv from the pipeline's outputs. Text blocks come from README.md."""
import sys;sys.path.insert(0,'models')
import numpy as np,pandas as pd,json,re,html
from scipy import stats
from core import unpack
e=html.escape
O=pd.read_csv('output/final_comparison.csv');D=pd.read_csv('output/race_table_full.csv');F=pd.read_csv('predictions.csv');F['race']=F.race_id.str[:2]
dup=(D.pd_user>=.5).values   # columns in O are oriented to an earlier label; re-orient everything to the pipeline's winner
pool_d=np.where(dup,O.pool,1-O.pool)
win_d=(pool_d>=.5)
same=dup==win_d
def ori(c):return np.where(same,O[c],1-O[c])
T=pd.DataFrame(dict(race=O.race,model=ori('model'),ratings=ori('ratings'),market=ori('market'),kalshi=ori('kalshi'),poly=ori('poly'),average=ori('pool'),
  lo=np.where(same,O.lo,1-O.hi),hi=np.where(same,O.hi,1-O.lo),miss=ori('R53'),polls=O.polls,n30=O.n30,margin=O.margin,lean=O.lean,kind=O.kind,rating_score=O.rate,receipts_dem_side=O.drcpt,receipts_rep=O.rrcpt)).merge(F[['race','candidate_name','predicted_winner_party','win_probability']],on='race')
assert (T.win_probability==np.floor(T.average*100+.5)).all()
# seats from unrounded averages
th=unpack(np.array(json.load(open('data/processed/full_fit.json'))['theta']));ind=(D.kind=='I').values;mg=np.where(ind,D.mu_raw-4.85,D.mu_raw)
r=np.random.default_rng(21);S=400000;nat=stats.t.rvs(8,size=S,random_state=r)*th['t_n']/np.sqrt(8/6);lf=r.normal(0,th['t_l'],S);dv={d:r.normal(0,th['t_r'],S) for d in np.unique(D['div'])};idio=stats.t.rvs(12,size=(S,len(D)),random_state=r)/np.sqrt(12/10);si=r.normal(0,4.5,S);s=np.zeros((S,len(D)))
for i,(z,dd,ki,v_) in enumerate(zip(D.z,D['div'],D.kind,D['var'])):
    x=mg[i]+nat+lf*z+dv[dd]+np.sqrt(v_)*idio[:,i]+(si if ki=='I' else 0);s[:,i]=x-np.quantile(x,1-np.clip(pool_d[i],1e-4,1-1e-4))
ds=(D.kind=='D').values;Ds=34+(s[:,ds]>0).sum(1);Iw=(s[:,~ds]>0).sum(1)
sd=pd.Series(Ds).value_counts(normalize=True).sort_index().rename('p');sd.to_csv('output/seat_distribution.csv')
SM=dict(mean=Ds.mean(),q10=np.quantile(Ds,.1),q90=np.quantile(Ds,.9),p51=(Ds>=51).mean(),p50=(Ds==50).mean(),ind=(Iw>0).mean());print({k:round(float(v),3) for k,v in SM.items()})
# README blocks
R=open('README.md',encoding='utf8').read()
def inl(t):
    t=e(t);t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t);t=re.sub(r'`(.+?)`',r'<code>\1</code>',t);return re.sub(r'\[(.+?)\]\((https?://[^)]+)\)',r'<a href="\2">\1</a>',t)
def md_table(block):
    L=[l for l in block.strip().splitlines() if l.startswith('|')];al=[c.strip().endswith(':') for c in L[1].strip('|').split('|')]
    row=lambda l:[c.strip() for c in l.strip().strip('|').split('|')]
    h=''.join(f'<th{" class=n" if a else ""}>{inl(c)}</th>' for c,a in zip(row(L[0]),al))
    b=''.join('<tr>'+''.join(f'<td{" class=n" if a else ""}>{inl(c)}</td>' for c,a in zip(row(l),al))+'</tr>' for l in L[2:])
    return f'<div class="scroll"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'
def tbl_after(head):
    i=R.index(head);m=re.search(r'((?:^\|.*\n)+)',R[i:],re.M);return md_table(m.group(1))
cont=re.findall(r'^- \*\*([A-Z]{2}), (.+?):\*\* (.+)$',R[R.index('### Contested races'):R.index('### Clear races')],re.M)
for st,lab,_ in cont:
    assert lab.endswith(str(int(T[T.race==st].win_probability.iloc[0]))),st
pc=lambda x:f"{x*100:.0f}%" if x<.995 else f"{x*100:.1f}%"
cards=''
for st,lab,txt in cont:
    t=T[T.race==st].iloc[0]
    cards+=f'<article class="race"><header><h3>{st} · {e(t.candidate_name)} ({t.predicted_winner_party[0]})</h3><span class="pill">{t.win_probability}%</span></header><p>{inl(txt)}</p><dl><div><dt>Model</dt><dd>{pc(t.model)}</dd></div><div><dt>Model range</dt><dd>{t.lo*100:.0f}–{t.hi*100:.0f}</dd></div><div><dt>Ratings</dt><dd>{pc(t.ratings)}</dd></div><div><dt>Market</dt><dd>{pc(t.market)}</dd></div><div><dt>Model if polls overstate Dems by 5.3</dt><dd>{pc(t.miss)}</dd></div></dl></article>'
# chart 1
C=T[T.race.isin([c[0] for c in cont])].sort_values('win_probability').reset_index(drop=True)
W,RH,L_,R_,Tp=780,38,190,62,28;H=Tp+RH*len(C)+36;x=lambda v:L_+(v-40)/60*(W-L_-R_)
c=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Win probability for each contested race from each source" class="chart">']
for t in (40,50,60,70,80,90,100):c.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{Tp-6}" y2="{H-30}" class="{"mid" if t==50 else "grid"}"/><text x="{x(t):.1f}" y="{H-12}" class="tick" text-anchor="middle">{t}%</text>')
c.append(f'<text x="{W-R_+12}" y="{Tp-12}" class="hd">Final</text>')
for i,t in C.iterrows():
    y=Tp+RH*i+RH/2-2;tip=f"{t.candidate_name}|Final {t.win_probability}%|Model {pc(t.model)} (range {t.lo*100:.0f}–{t.hi*100:.0f})|Ratings {pc(t.ratings)}|Market {pc(t.market)}"
    c.append(f'<g class="row" tabindex="0" data-tip="{e(tip)}"><rect x="0" y="{y-RH/2:.1f}" width="{W}" height="{RH}" class="hit"/><text x="8" y="{y+4:.1f}" class="lab"><tspan class="st">{t.race}</tspan> {e(t.candidate_name.split()[-1])} ({t.predicted_winner_party[0]})</text>')
    c.append(f'<line x1="{x(t.lo*100):.1f}" x2="{x(t.hi*100):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="band"/>')
    a=x(t.market*100);c.append(f'<path d="M{a:.1f} {y-5.5:.1f} L{a+5.5:.1f} {y+4.5:.1f} L{a-5.5:.1f} {y+4.5:.1f} Z" class="m3"/>')
    a=x(t.ratings*100);c.append(f'<rect x="{a-4.5:.1f}" y="{y-4.5:.1f}" width="9" height="9" class="m2"/><circle cx="{x(t.model*100):.1f}" cy="{y:.1f}" r="5" class="m1"/>')
    a=x(t.win_probability);c.append(f'<path d="M{a:.1f} {y-8:.1f} L{a+8:.1f} {y:.1f} L{a:.1f} {y+8:.1f} L{a-8:.1f} {y:.1f} Z" class="you"/><text x="{W-R_+12}" y="{y+4:.1f}" class="val">{t.win_probability}</text></g>')
c.append('</svg>');C1=''.join(c)
# chart 2
Cal=pd.read_csv('output/calibration_backtest.csv');W2,H2,L2,B2=420,340,46,40;xx=lambda v:L2+(v-50)/50*(W2-L2-14);yy=lambda v:H2-B2-(v-30)/70*(H2-B2-14)
c=[f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-label="Predicted probability against actual win rate in the Senate backtest" class="chart small-chart">']
for t in (50,60,70,80,90,100):c.append(f'<line x1="{xx(t):.1f}" x2="{xx(t):.1f}" y1="14" y2="{H2-B2}" class="grid"/><text x="{xx(t):.1f}" y="{H2-B2+18}" class="tick" text-anchor="middle">{t}</text>')
for t in (40,60,80,100):c.append(f'<line x1="{L2}" x2="{W2-14}" y1="{yy(t):.1f}" y2="{yy(t):.1f}" class="grid"/><text x="{L2-8}" y="{yy(t)+4:.1f}" class="tick" text-anchor="end">{t}</text>')
c.append(f'<line x1="{xx(50):.1f}" y1="{yy(50):.1f}" x2="{xx(100):.1f}" y2="{yy(100):.1f}" class="diag"/><text x="{xx(62):.1f}" y="{yy(46):.1f}" class="tick">diagonal = perfect calibration</text>')
for t in Cal.itertuples():
    tip=f"{t.fav}|Predicted {t.pred*100:.1f}%|Actual {t.obs*100:.1f}%|{t.n} races"
    c.append(f'<g class="row" tabindex="0" data-tip="{e(tip)}"><circle cx="{xx(t.pred*100):.1f}" cy="{yy(t.obs*100):.1f}" r="16" class="hit"/><circle cx="{xx(t.pred*100):.1f}" cy="{yy(t.obs*100):.1f}" r="{np.clip(3+np.sqrt(t.n)*.55,4,13):.1f}" class="m1s"/></g>')
c.append(f'<text x="{(L2+W2-14)/2:.0f}" y="{H2-6}" class="tick" text-anchor="middle">Forecast probability for the favorite (%)</text></svg>');C2=''.join(c)
caltbl=''.join(f'<tr><td class=nw>{t.fav}</td><td class=n>{t.n}</td><td class=n>{t.pred*100:.1f}%</td><td class=n>{t.obs*100:.1f}%</td></tr>' for t in Cal.itertuples())
# chart 3
seats=list(range(46,59));W3,H3=780,230;bw=(W3-30)/len(seats);ym=sd.max();c=[f'<svg viewBox="0 0 {W3} {H3}" role="img" aria-label="Distribution of Democratic Senate seats" class="chart">']
for i,v in enumerate(seats):
    p=float(sd.get(v,0));h=p/ym*(H3-66);X=15+i*bw;cl='m1s' if v>=51 else 'neu'
    c.append(f'<g class="row" tabindex="0" data-tip="{v} Democratic seats|{p*100:.1f}% of simulations"><rect x="{X:.1f}" y="10" width="{bw:.1f}" height="{H3-46}" class="hit"/><rect x="{X+2:.1f}" y="{H3-36-h:.1f}" width="{bw-4:.1f}" height="{max(h,1):.1f}" rx="3" class="{cl}"/><text x="{X+bw/2:.1f}" y="{H3-18}" class="tick" text-anchor="middle">{v}</text>'+(f'<text x="{X+bw/2:.1f}" y="{H3-42-h:.1f}" class="val" text-anchor="middle">{p*100:.0f}%</text>' if p>=.02 else '')+'</g>')
c.append('</svg>');C3=''.join(c)
# reference classes
Rc=pd.read_csv('output/reference_classes.csv');rc=''.join(f'<tr><td>{e(t.cls)}</td><td class=n>{t.n}</td><td class=n>{t.cycles}</td><td class=n>{t.won*100:.0f}%</td><td class=n>{t.lo*100:.0f}–{t.hi*100:.0f}</td></tr>' for t in Rc.itertuples())
# ledger
basis={c_[0]:c_[2] for c_ in cont}
for l in R[R.index('### Clear races'):R.index('## Validation')].splitlines():
    m=re.match(r'^\| ([A-Z]{2}) \| (.+?) \| [\d.]+ \|$',l)
    if m:basis[m.group(1)]=m.group(2)
T['basis']=T.race.map(basis);assert T.basis.notna().all()
T.drop(columns=['receipts_dem_side','receipts_rep']).assign(receipts_dem_side=O.drcpt.values,receipts_rep=O.rrcpt.values).rename(columns=dict(win_probability='final',lo='model_range_lo',hi='model_range_hi',miss='model_if_polls_overstate_dems_by_5_3',candidate_name='winner',predicted_winner_party='party')).round(4).to_csv('output/justification_by_race.csv',index=False)
tpl=open('models/report_template.html',encoding='utf8').read();style=tpl[:tpl.index('<main>')].replace('<title>Senate Forecast Audit</title>','<title>2026 Senate Forecast</title>');script=tpl[tpl.index('<div id="tip"'):]
body=open('models/report_body.html',encoding='utf8').read()
pred_tbl=tbl_after('## Predictions');clear=tbl_after('### Clear races')
for k,v in dict(C1=C1,C2=C2,C3=C3,CARDS=cards,CALTBL=caltbl,RC=rc,PRED=pred_tbl,CLEAR=clear,SETTINGS=tbl_after('### 2. Poll average'),GOV=tbl_after('### 6. Calibration'),MARKET=tbl_after('### 8. Market prices'),DATA=tbl_after('### 1. Data'),MEAN=f"{SM['mean']:.1f}",Q10=f"{SM['q10']:.0f}",Q90=f"{SM['q90']:.0f}",P51=f"{SM['p51']*100:.0f}",P50=f"{SM['p50']*100:.0f}",IND=f"{SM['ind']*100:.0f}",DWINS=str(int((F.predicted_winner_party=='Democratic').sum())),RWINS=str(int((F.predicted_winner_party=='Republican').sum())),DSEATS=str(34+int(((F.predicted_winner_party=='Democratic')).sum()))).items():
    body=body.replace('{{'+k+'}}',v)
assert not re.findall(r'\{\{\w+\}\}',body),re.findall(r'\{\{\w+\}\}',body)
open('output/report.html','w',encoding='utf8').write(style+body+script)
print('ok',len(style+body+script),'contested',len(cont),'basis',T.basis.notna().sum(),F.predicted_winner_party.value_counts().to_dict(),F.predicted_winner_party.isin(['Independent']).sum())
for w in ('original','Your ','your ','Proposed','proposed'):print(w,(style+body).count(w))
