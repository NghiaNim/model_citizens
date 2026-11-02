"""Builds output/report.html from output/*.csv. Every number comes from the pipeline except hand-written race notes."""
import pandas as pd,numpy as np,json,html
from scipy import stats,optimize
O=pd.read_csv('output/final_comparison.csv');Rc=pd.read_csv('output/reference_classes.csv');Cal=pd.read_csv('output/calibration_backtest.csv');Seat=pd.read_csv('output/seat_distribution.csv',index_col=0);SS=json.load(open('output/seat_summary.json'));ES=pd.read_csv('output/expected_score.csv')
k,kT=1.051,1.051
def logit(p):p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
def expit(x):return 1/(1+np.exp(-x))
def implied(r):
    """poll miss (points toward the pick's opponent, + = pick does worse than model mean) needed for model to equal user's number"""
    kk=1. if r.kind=='I' else (kT if r.trailing else k); s=1 if O.loc[O.race==r.race,'margin'].iloc[0]*1>=0 else -1
    pick_m=abs(r.margin) if r.model>=.5 else -abs(r.margin)
    f=lambda sh:expit(kk*logit(stats.norm.cdf((pick_m-sh)/r.sd)))-r.user/100
    try:return optimize.brentq(f,-30,30)
    except Exception:return np.nan
O['implied']=O.apply(implied,axis=1)
pct=lambda x:f"{x*100:.0f}" if .005<x<.995 else (f"{x*100:.1f}")
pc=lambda x:f"{x*100:.0f}%" if x<.995 else f"{x*100:.1f}%"
NOTES={
'MI':("Raise to 87","El-Sayed's weighted lead is 4.9 over 19 recent polls, 5.6 in the last 10 days, including R-sponsored Rasmussen at +5. Leads of 4 to 6 points held in 26 of 27 Senate races and 23 of 28 governor races, 89% pooled. It's an open seat, and ratings alone give 92%. The market's 78 looks low; my earlier 93 looked too high."),
'SC':("Raise to 83","This is the widest gap with the market and the raise I'd push least hard. Graham leads by 1.6 on average and 0.8 in the last 10 days, with NYT/Siena tied at 48 and a D-sponsored poll at Andrews +5. Ratings (Lean to Likely R) give 93%, and narrow Republican leaders in red states won 94% of the time in Senate races (n=31). The case against: she's an appointee with no record as a candidate, and the polls have moved 3.6 points toward Andrews over two months."),
'FL':("Raise to 97","Moody has led by 6 across 19 polls, stable in all three windows. Every lead of 6 to 8 points held in both data sets (51 of 51), and her adjusted margin of 7.3 is larger than any Senate cycle's average polling error (the largest was 5.3, in 2020)."),
'MS':("Raise to 95","Only four public polls exist, all D-sponsored, and every one has Hyde-Smith ahead, by 3 to 13. Sponsor bias averages 2.1 points. Ratings average Likely-to-Safe R, and Mississippi polls have missed against Democrats by a mean of 4.5. Pinkins (I) polls at about 6%."),
'TX':("Keep 75","Talarico leads Paxton by 3.4 in 32 polls from the last 30 days, the deepest polling in the country, and he has raised $146M to Paxton's $22M. But leads of 2 to 4 points held only 75% of the time in Senate races and 60% in governor races, and Texas polls missed against Democrats by 4 in both 2020 and 2024. The three-source average is 78, too close to your 75 to justify a change. I earlier proposed 84 and withdrew it after the governor test."),
'LA':("Keep 88","Only three polls exist, all D-sponsored, with a tie in the latest (Hart, Oct 28 to 29). Ratings give 99.6, but without the Senate-only sharpening the polls alone give 81, so the sources now sit on opposite sides of your number and I withdrew my earlier 97. The market has 86.5, from Polymarket only since no Kalshi Louisiana market was on the feed."),
'IA':("Keep 52, but it's a coin flip","Polls are +0.8 for Turek, 0.0 in the last 10 days. Undecideds historically break toward the state's lean, which erases the lead (model 49.5%). The market is at 50 and ratings at 28, with Sabato's final call Leans R. So nothing supports Turek as a real favorite, and nothing rules him out. My rule doesn't flip a pick on this evidence, but this is your least-supported call."),
'KS':("Keep 63","Model 74 (range 69 to 82), market 62, ratings 56, and the three average 64. Hamilton's lead has grown from -0.1 to +1.4 to +4.6 across windows, with NYT/Siena at +5. Against that, Marshall is a trailing incumbent in a state about 16 points redder than the nation, and Kansas polls have missed against Democrats by 2.7 on average."),
'OH':("Keep 71","Model 70 (range 66 to 77), ratings 78, market 72. This is the best-supported of your competitive numbers. Brown leads by 2.9 over 20 polls but has drifted down (+4.1, +2.8, +2.4), and Husted is an appointee."),
'AK':("Keep 72, leaning high","Ratings (72) and the market (72.5) agree with you, but polls alone give 62 (range 59 to 67), so your number is above the model's range. Alaska's polls missed against Democrats in all five cycles on record, by 4.4 on average. Peltola led the August primary 49.5 to 41.4, and final-round polls have her up about 2. If you moved this, 69 is where the combined evidence sits."),
'ME':("Keep 73, leaning low","Jackson's lead has widened (+1.8, +2.6, +5.2), which pulls the model to 86 and ratings to 85; your 73 is below the model's range (80 to 93). I'm not moving you, because the same state and incumbent produced the largest Senate miss in this data: Collins beat her polls by about 13 in 2020. Challengers leading incumbents won 18 of 26 pooled Senate and governor races (69%), and that history supports staying near your number."),
'NE':("Keep 71, leaning low","Run the model on polls at face value and it returns about 69, close to your number. The dispute is one assumption. Independents have tended to run behind their polls: Orman 2014 by 12, McMullin 2022 by 3 to 5, Osborn 2024 by 4, though Walker 2014 ran 1 ahead. Applying that gives 90 for Ricketts, and ratings give 96. Four races are too few to overrule you, and Osborn lost by 6.7 in a far worse national environment than this one. But the evidence leans toward Ricketts above 71."),
}
SAFE={'GA':'Ossoff +10. A Dec 1 runoff needs both candidates under 50%: about a 2% chance of a margin within 2 points. A 5.3-point polling miss drops this to 87%.','NC':'Cooper +10 across 11 recent polls. Ratings range Lean to Safe D. A 5.3-point polling miss drops this to 87%.','NH':'Pappas +9, but Sununu is a real opponent and Cook has Lean D. Governor races at this lead are less secure than Senate ones.','MN':'Flanagan +6.5, and Cook still has Lean D. A 5.3-point polling miss drops this to 61%, the weakest of your 100s.','MT':'Three-way: Bankhead (D) polls at 10 to 23%, splitting the anti-Alme vote. D-sponsored polls show a tie and R-sponsored ones about R+16, so the average is unstable.','ID':'Achilles (I): polls range from R+25 to I+11, the widest spread here.','SD':'Rounds +11 over Bengs (I). Recent polls range from Bengs +2 (Concord) to R+21 (Quantus).','KY':'No polls in 60 days; the last two were R+9 and R+11. Kentucky runs 29 points redder than the nation. Rests on fundamentals and ratings.','OK':'No polls in 60 days. Oklahoma runs 34 points redder than the nation; Hern has raised $10.9M against $116K.','AR':'Cotton +9, on just two polls, with Shoffner raising $4M. Ratings all Safe.'}
def safe_note(r):
    if r.race in SAFE:return SAFE[r.race]
    if r.n30==0 and pd.isna(r.polls):return f"No public polls in 60 days. The state runs {abs(r.lean):.0f} points more {'Democratic' if r.lean>0 else 'Republican'} than the nation; every rater Safe."
    return f"Polling {abs(r.polls):.0f} points ahead on {int(r.n30)} recent poll{'s' if r.n30!=1 else ''}; every rater Safe or Likely."
e=html.escape
# ---- Chart 1
comp=O[O.user<100].sort_values('user').reset_index(drop=True)
W,RH,L,R_,T=780,38,178,70,28;H=T+RH*len(comp)+36;x=lambda v:L+(v-20)/(100-20)*(W-L-R_)
s=[f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Probability of each predicted winner from four sources, twelve competitive races" class="chart">']
for t in (20,40,60,80,100):
    s.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T-6}" y2="{H-30}" class="grid"/><text x="{x(t):.1f}" y="{H-12}" class="tick" text-anchor="middle">{t}%</text>')
s.append(f'<line x1="{x(50):.1f}" x2="{x(50):.1f}" y1="{T-6}" y2="{H-30}" class="mid"/>')
s.append(f'<text x="{W-R_+10}" y="{T-12}" class="hd">Proposed</text>')
for i,r in comp.iterrows():
    y=T+RH*i+RH/2-2
    tip=f"{r.cand}|Yours {r.user}%|Model {pc(r.model)} (range {r.lo*100:.0f}–{r.hi*100:.0f})|Ratings {pc(r.ratings)}|Market {pc(r.market)}|Proposed {r.proposed}%"
    s.append(f'<g class="row" tabindex="0" data-tip="{e(tip)}"><rect x="0" y="{y-RH/2:.1f}" width="{W}" height="{RH}" class="hit"/>')
    s.append(f'<text x="8" y="{y+4:.1f}" class="lab"><tspan class="st">{r.race}</tspan> {e(r.cand.split()[-1])}</text>')
    s.append(f'<line x1="{x(max(r.lo*100,20)):.1f}" x2="{x(r.hi*100):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="band"/>')
    tx=x(r.market*100);s.append(f'<path d="M{tx:.1f} {y-5.5:.1f} L{tx+5.5:.1f} {y+4.5:.1f} L{tx-5.5:.1f} {y+4.5:.1f} Z" class="m3"/>')
    rx=x(r.ratings*100);s.append(f'<rect x="{rx-4.5:.1f}" y="{y-4.5:.1f}" width="9" height="9" class="m2"/>')
    s.append(f'<circle cx="{x(r.model*100):.1f}" cy="{y:.1f}" r="5" class="m1"/>')
    ux=x(r.user);s.append(f'<path d="M{ux:.1f} {y-8:.1f} L{ux+8:.1f} {y:.1f} L{ux:.1f} {y+8:.1f} L{ux-8:.1f} {y:.1f} Z" class="you"/>')
    changed=r.proposed!=r.user
    s.append(f'<text x="{W-R_+10}" y="{y+4:.1f}" class="val{" chg" if changed else ""}">{r.proposed}{"" if changed else " ="}</text></g>')
s.append('</svg>');C1=''.join(s)
# ---- Chart 2 calibration
W2,H2,L2,B2=420,340,46,40;xx=lambda v:L2+(v-50)/50*(W2-L2-14);yy=lambda v:H2-B2-(v-30)/70*(H2-B2-14)
s=[f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-label="Predicted probability against observed win rate in the held-out backtest" class="chart small-chart">']
for t in (50,60,70,80,90,100):s.append(f'<line x1="{xx(t):.1f}" x2="{xx(t):.1f}" y1="14" y2="{H2-B2}" class="grid"/><text x="{xx(t):.1f}" y="{H2-B2+18}" class="tick" text-anchor="middle">{t}</text>')
for t in (40,60,80,100):s.append(f'<line x1="{L2}" x2="{W2-14}" y1="{yy(t):.1f}" y2="{yy(t):.1f}" class="grid"/><text x="{L2-8}" y="{yy(t)+4:.1f}" class="tick" text-anchor="end">{t}</text>')
s.append(f'<line x1="{xx(50):.1f}" y1="{yy(50):.1f}" x2="{xx(100):.1f}" y2="{yy(100):.1f}" class="diag"/><text x="{xx(79):.1f}" y="{yy(87):.1f}" class="tick">perfect calibration</text>')
for r in Cal.itertuples():
    se=np.sqrt(max(r.obs*(1-r.obs),.02)/r.n)*100
    s.append(f'<g class="row" tabindex="0" data-tip="{e(f"{r.fav}|Predicted {r.pred*100:.1f}%|Observed {r.obs*100:.1f}%|{r.n} races")}"><circle cx="{xx(r.pred*100):.1f}" cy="{yy(r.obs*100):.1f}" r="16" class="hit"/><circle cx="{xx(r.pred*100):.1f}" cy="{yy(r.obs*100):.1f}" r="{np.clip(3+np.sqrt(r.n)*.55,4,13):.1f}" class="m1s"/></g>')
s.append(f'<text x="{(L2+W2-14)/2:.0f}" y="{H2-6}" class="tick" text-anchor="middle">Forecast probability for the favorite (%)</text></svg>');C2=''.join(s)
# ---- Chart 3 seats
sr=Seat.p;seats=list(range(46,59));W3,H3=780,230;bw=(W3-30)/len(seats);ym=sr.max();
s=[f'<svg viewBox="0 0 {W3} {H3}" role="img" aria-label="Distribution of Democratic Senate seats" class="chart">']
for i,v in enumerate(seats):
    p=float(sr.get(v,0));h=p/ym*(H3-66);X=15+i*bw;cl='m1s' if v>=51 else 'neu'
    s.append(f'<g class="row" tabindex="0" data-tip="{v} Democratic seats|{p*100:.1f}% of simulations"><rect x="{X:.1f}" y="10" width="{bw:.1f}" height="{H3-46}" class="hit"/><rect x="{X+2:.1f}" y="{H3-36-h:.1f}" width="{bw-4:.1f}" height="{max(h,1):.1f}" rx="3" class="{cl}"/><text x="{X+bw/2:.1f}" y="{H3-18}" class="tick" text-anchor="middle">{v}</text>'+(f'<text x="{X+bw/2:.1f}" y="{H3-42-h:.1f}" class="val" text-anchor="middle">{p*100:.0f}%</text>' if p>=.02 else '')+'</g>')
s.append('</svg>');C3=''.join(s)
# ---- tables
_B=pd.read_csv('data/processed/backtest_final.csv');_p=expit(1.051*logit(stats.norm.cdf(_B.mu/_B.sd)));_y=(_B.margin>0)*1.
print('2024 Brier %.3f misses %s'%(np.mean((_p-_y)[_B.year==2024]**2),list(_B[(_B.year==2024)&((_p>.5)!=(_B.margin>0))].st)))
_C=pd.read_csv('data/raw/fte/co_us_senate_elections.csv');_C=_C[_C.party=='D'].copy();_C['special']=_C.special.astype(str).str.lower().eq('true');_C['forecast_date']=pd.to_datetime(_C.forecast_date);_C=_C[_C.forecast_type=='deluxe'].sort_values('forecast_date').groupby(['year','state','special']).tail(1)
_m=_B.assign(pp=_p).merge(_C[['year','state','special','probwin','probwin_outcome']],left_on=['year','st','special'],right_on=['year','state','special']);_o=_m.probwin_outcome.astype(float)
print('538 same races n=%d 538 %.4f mine %.4f'%(len(_m),np.mean((_m.probwin-_o)**2),np.mean((_m.pp-_o)**2)))
_rt=pd.read_csv('data/processed/ratings_hist.csv');_mm=_B.assign(pp=_p).merge(_rt,on=['year','st'],suffixes=('','_r'));print('vs ratings same races n=%d mine %.4f ratings %.4f'%(len(_mm),np.mean((_mm.pp-(_mm.margin>0))**2),np.mean((1/(1+np.exp(-1.642*_mm.score))-(_mm.margin>0))**2)))
chg=O[(O.user!=O.proposed)&(O.user<100)].sort_values('proposed')
imp=''.join(f'<tr><td>{r.race} {e(r.cand.split()[-1])}</td><td class=n>{r.user}</td><td class=n>{r.proposed}</td><td class=n>{pc(r.model)}</td><td class=n>{pc(r.ratings)}</td><td class=n>{pc(r.market)}</td><td class=n>{(r.user+r.proposed)/2:.1f}%</td><td class=n>{r.implied:+.1f}</td></tr>' for r in O[(O.user!=O.proposed)&(O.user<100)].sort_values('user',ascending=False).itertuples())
comp_cards=''
order=['MI','SC','FL','MS','TX','LA','IA','KS','OH','AK','ME','NE']
for st in order:
    r=O[O.race==st].iloc[0];v,t=NOTES[st];ch=r.proposed!=r.user
    comp_cards+=f'<article class="race"><header><h3>{st} · {e(r.cand)}</h3><span class="{"pill up" if ch else "pill"}">{e(v)}</span></header><p>{e(t)}</p><dl><div><dt>Model</dt><dd>{pc(r.model)}</dd></div><div><dt>Model range</dt><dd>{r.lo*100:.0f}–{r.hi*100:.0f}</dd></div><div><dt>Ratings</dt><dd>{pc(r.ratings)}</dd></div><div><dt>Market</dt><dd>{pc(r.market)}</dd></div><div><dt>Polls overstate Dems by 5.3</dt><dd>{pc(r.R53)}</dd></div></dl></article>'
safe=O[O.user==100].sort_values(['proposed','pool'])
safe_rows=''.join(f'<tr><td class=nw>{r.race} {e(r.cand.split()[-1])}</td><td class=n>{r.proposed}</td><td class=n>{pc(r.model)}</td><td class=n>{pc(r.ratings)}</td><td class=n>{pc(r.market)}</td><td class=wr>{e(safe_note(r))}</td></tr>' for r in safe.itertuples())
rc=''.join(f'<tr><td>{e(r.cls)}</td><td class=n>{r.n}</td><td class=n>{r.cycles}</td><td class=n>{r.won*100:.0f}%</td><td class=n>{r.lo*100:.0f}–{r.hi*100:.0f}</td></tr>' for r in Rc.itertuples())
pool_p=SS['proposed']['p51'];u_p=SS['user']['p51'];es=ES.set_index('truth')

# ---- every race: justification ledger, including what reproduces the original number
def support(r):
    srcs=[('model',r.model),('ratings',r.ratings),('market',r.market)]
    near=[n for n,v in srcs if abs(v*100-r.user)<4]
    return ', '.join(near) if near else 'none within 4 points'
def basis_txt(r):
    return NOTES[r.race][1] if r.race in NOTES else safe_note(r)
L_=O.copy()
L_['sources_within_4_of_original']=L_.apply(support,axis=1)
L_['extra_poll_miss_to_reproduce_original']=L_.implied.round(2)
L_['basis']=L_.apply(basis_txt,axis=1)
L_[['race','cand','user','proposed','decision','model','lo','hi','ratings','market','pool','polls','n30','neff','margin','sd','rate','lean','money','inc','kind','trailing','sources_within_4_of_original','extra_poll_miss_to_reproduce_original','R53','D2','tau7','tau28','noind','basis']].rename(columns=dict(user='original',lo='model_band_lo',hi='model_band_hi',pool='three_source_average',R53='if_polls_overstate_D_by_5_3',D2='if_polls_understate_D_by_2')).round(4).to_csv('output/justification_by_race.csv',index=False)
kept=O[(O.user<100)&(O.user==O.proposed)].sort_values('user')
repro=''.join(f'<tr><td>{r.race} {e(r.cand.split()[-1])}</td><td class=n>{r.user}</td><td class=n>{pc(r.model)}</td><td class=n>{r.lo*100:.0f}–{r.hi*100:.0f}</td><td class=n>{r.implied:+.1f}</td><td>{e(support(r))}</td></tr>' for r in kept.itertuples())
page=open('models/report_template.html',encoding='utf8').read()
caltbl=''.join(f'<tr><td class=nw>{r.fav}</td><td class=n>{r.n}</td><td class=n>{r.pred*100:.1f}%</td><td class=n>{r.obs*100:.1f}%</td></tr>' for r in Cal.itertuples())
page=page.replace('<tbody id="caltbl"></tbody>','<tbody>'+caltbl+'</tbody>')
for a,b in dict(REPRO=repro,C1=C1,C2=C2,C3=C3,IMP=imp,CARDS=comp_cards,SAFE=safe_rows,RC=rc,PROPMEAN=f"{SS['proposed']['mean']:.1f}",PROP51=f"{pool_p*100:.0f}",USER51=f"{u_p*100:.0f}",MODEL51=f"{SS['model']['p51']*100:.0f}",MK51=f"{SS['market']['p51']*100:.0f}",P50=f"{SS['proposed']['p50']*100:.0f}",Q10=f"{SS['proposed']['q10']:.0f}",Q90=f"{SS['proposed']['q90']:.0f}",B_MODEL_U=f"{es.brier_user['model']:.2f}",B_MODEL_P=f"{es.brier_prop['model']:.2f}",B_RAT_U=f"{es.brier_user['ratings']:.2f}",B_RAT_P=f"{es.brier_prop['ratings']:.2f}",B_MK_U=f"{es.brier_user['market']:.2f}",B_MK_P=f"{es.brier_prop['market']:.2f}",B_PO_U=f"{es.brier_user['pool']:.2f}",B_PO_P=f"{es.brier_prop['pool']:.2f}").items():
    page=page.replace('{{'+a+'}}',b)
open('output/report.html','w',encoding='utf8').write(page)
print(O[O.user<100][['race','user','proposed','implied','model','R53','D2']].round(2).to_string())
import re;print('unfilled',re.findall(r'\{\{\w+\}\}',page))
