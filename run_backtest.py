import duckdb, numpy as np, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import pipeline as P

con=duckdb.connect()
TEST=[2018,2019,2020,2021,2022,2023,2024,2025]
NEW=['rz_csh','gl_csh','rz_tsh','implied']+P.NGS

def features(con):
    df=con.sql("""select w.season,w.week,w.posteam,w.ppos,w.carries,w.targets,w.scored,
        w.c_rush,w.c_rec,w.c_car,w.c_tgt,w.c_gl,w.c_rzr,w.c_ezt,w.c_rzt,w.c_scr,w.c_g,w.a_snap,
        w.c_car_l3,w.c_tgt_l3,w.c_gl_l3,w.c_g_l3,w.a_snap_l3,
        pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g,
        tr.tgl_tr,tr.trz_tr,tr.trt_tr, ve.implied,
        nt.n_sep,nt.n_yacoe,nt.n_cush,nt.n_catchp,nt.n_eff,nt.n_ryoe,nt.n_box8,nt.n_rpoe,
        np.p_sep,np.p_yacoe,np.p_cush,np.p_catchp,np.p_eff,np.p_ryoe,np.p_box8,np.p_rpoe
      from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season
      left join teamrz_trail tr on w.season=tr.season and w.week=tr.week and w.posteam=tr.posteam
      left join vegas ve on w.game_id=ve.game_id and w.posteam=ve.team
      left join ngs_trail nt on w.pid=nt.pid and w.season=nt.season and w.week=nt.week
      left join ngs_prior np on w.pid=np.pid and w.season=np.season""").fetchnumpy()
    return df

def c0(a): a=np.asarray(a,float); a[np.isnan(a)]=0.0; return a
def make(df, trail):
    season=np.asarray(df['season']);week=np.asarray(df['week']);pos=df['ppos'].astype(str)
    team=np.asarray(df['posteam'])
    has_pr=c0(df['pr_g'])>0; c_g=c0(df['c_g'])
    gm={n:float(np.mean(c0(df[n])[has_pr])) for n in ['pr_rush','pr_rec','pr_car','pr_tgt','pr_gl','pr_rzr','pr_ezt','pr_rzt','pr_scr']}
    a_snap=np.asarray(df['a_snap'],float); pr_snap=np.asarray(df['pr_snap'],float)
    gsnap=float(np.nanmean(a_snap[~np.isnan(a_snap)]))
    def sh(cs,pp,gl): return (c0(cs)+K*np.where(has_pr,c0(pp),gl))/(c_g+K)
    K=P.K
    xr=sh(df['c_rush'],df['pr_rush'],gm['pr_rush']); xc=sh(df['c_rec'],df['pr_rec'],gm['pr_rec'])
    cpg=sh(df['c_car'],df['pr_car'],gm['pr_car']); tpg=sh(df['c_tgt'],df['pr_tgt'],gm['pr_tgt'])
    glp=sh(df['c_gl'],df['pr_gl'],gm['pr_gl']); rzr=sh(df['c_rzr'],df['pr_rzr'],gm['pr_rzr'])
    ezt=sh(df['c_ezt'],df['pr_ezt'],gm['pr_ezt']); rzt=sh(df['c_rzt'],df['pr_rzt'],gm['pr_rzt'])
    nv=sh(df['c_scr'],df['pr_scr'],gm['pr_scr']); xtd=xr+xc; vol=cpg+tpg
    snap=np.where(~np.isnan(a_snap),a_snap,np.where(~np.isnan(pr_snap),pr_snap,gsnap))
    # NEW carry/target shares (current-season trailing)
    trz=c0(df['trz_tr']); tgl=c0(df['tgl_tr']); trt=c0(df['trt_tr'])
    rz_csh=np.clip((c0(df['c_gl'])+c0(df['c_rzr']))/np.maximum(trz,1e-6),0,1.2)
    gl_csh=np.clip(c0(df['c_gl'])/np.maximum(tgl,1e-6),0,1.2)
    rz_tsh=np.clip((c0(df['c_ezt'])+c0(df['c_rzt']))/np.maximum(trt,1e-6),0,1.2)
    implied=np.asarray(df['implied'],float); implied[np.isnan(implied)]=22.0
    team_exp=np.array([trail.get((f"{int(season[i])}_{team[i]}",int(week[i])),2.4) for i in range(len(season))])
    g3=c0(df['c_g_l3'])
    def r3(cs,fb): v=c0(cs)/np.maximum(g3,1e-6); return np.where(g3>=1,v,fb)
    cpg3=r3(df['c_car_l3'],cpg);tpg3=r3(df['c_tgt_l3'],tpg);glp3=r3(df['c_gl_l3'],glp)
    a3=np.asarray(df['a_snap_l3'],float); snap3=np.where(~np.isnan(a3),a3,snap)
    trend_car=cpg3-cpg; trend_gl=glp3-glp; trend_tgt=tpg3-tpg
    ngs=P.ngs_cols(df, P.ngs_global_means(df))
    cols={'xr':xr,'xc':xc,'xtd':xtd,'vol':vol,'cpg':cpg,'tpg':tpg,'glpg':glp,'rzrpg':rzr,'eztpg':ezt,'rztpg':rzt,
          'rz_csh':rz_csh,'gl_csh':gl_csh,'rz_tsh':rz_tsh,'exp_gl_td':glp*0.38+rzr*0.08,'implied':implied,'team_exp':team_exp,'snap':snap,
          'naive':nv,'cg':c_g,'cpg3':cpg3,'tpg3':tpg3,'glp3':glp3,'snap3':snap3,'trend_car':trend_car,'trend_gl':trend_gl,'trend_tgt':trend_tgt,
          **ngs,
          'is_RB':(pos=='RB'),'is_WR':(pos=='WR'),'is_TE':(pos=='TE'),'is_QB':(pos=='QB')}
    X=np.column_stack([cols[f] for f in P.FEATS]).astype(float)
    Xbase=np.column_stack([cols[f] for f in P.FEATS if f not in NEW]).astype(float)
    carries=c0(df['carries']);targets=c0(df['targets']);scored=c0(df['scored']).astype(int)
    skill=np.isin(pos,['RB','WR','TE','QB']); base=skill&((carries+targets)>=1)&((c_g>=1)|has_pr)
    return X,Xbase,scored,base,season,pos,vol

def fitpred(Xtr,ytr,Xte):
    clf=HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,
        l2_regularization=1.0,min_samples_leaf=60,random_state=0,early_stopping=False)  # raw: well-calibrated at top end
    clf.fit(Xtr,ytr); return clf.predict_proba(Xte)[:,1]

def brier(y,p):return np.mean((p-y)**2)
def logloss(y,p):p=np.clip(p,1e-6,1-1e-6);return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
def auc(y,p):
    y=np.asarray(y);n1=y.sum();n0=len(y)-n1
    if n1==0 or n0==0:return float('nan')
    o=np.argsort(p,kind='mergesort');ra=np.empty(len(p),float);sp=p[o];i=0;r=1
    while i<len(sp):
        j=i
        while j+1<len(sp) and sp[j+1]==sp[i]:j+=1
        ra[o[i:j+1]]=(r+r+(j-i))/2.0;r+=(j-i+1);i=j+1
    return (ra[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def ece(y,p,b=10):
    e=np.quantile(p,np.linspace(0,1,b+1));e[0]-=1e-9;e[-1]+=1e-9;s=0.0
    for i in range(b):
        m=(p>e[i])&(p<=e[i+1])
        if m.sum():s+=m.sum()/len(p)*abs(p[m].mean()-y[m].mean())
    return s

RES={'v3':{}, 'base':{}, 'naive':{}}
POS={}; VOL={}
for T in TEST:
    P.build_tables(con, [y for y in range(2016,T+1)], [y for y in range(2016,T)])
    trail=P.team_env(con); df=features(con); X,Xbase,y,base,season,pos,vol=make(df,trail)
    tr=base&(season<T); te=base&(season==T)
    POS[T]=pos[te]; VOL[T]=vol[te]
    RES['v3'][T]=(y[te], fitpred(X[tr],y[tr],X[te]))
    RES['base'][T]=(y[te], fitpred(Xbase[tr],y[tr],Xbase[te]))
    # naive = trailing TD rate (feature 'naive' index)
    nv=X[:,P.FEATS.index('naive')]; RES['naive'][T]=(y[te], nv[te])
    print("done",T,"train",int(tr.sum()),"test",int(te.sum()))

def pool(d): return np.concatenate([d[s][0] for s in TEST]), np.concatenate([d[s][1] for s in TEST])
print("\n%-26s%8s%9s%7s%7s"%("model","Brier","LogLoss","AUC","ECE"))
for name in ['naive','base','v3']:
    y,p=pool(RES[name]); print("%-26s%8.4f%9.4f%7.3f%7.3f"%(
        {'naive':'Naive: past TD rate','base':'GBM (no carry-share/Vegas)','v3':'GBM + carry-share + Vegas'}[name],
        brier(y,p),logloss(y,p),auc(y,p),ece(y,p)))
print("\nPer season (v3):  %-6s %7s %7s %6s"%("season","Brier","LogLoss","AUC"))
for s in TEST:
    y,p=RES['v3'][s]; print("                  %-6d %7.4f %7.4f %6.3f"%(s,brier(y,p),logloss(y,p),auc(y,p)))

# dump v3 backtest json for the board
y,p=pool(RES['v3'])
def tier(pr): return 'Elite (45%+)' if pr>=.45 else 'Strong (33-45)' if pr>=.33 else 'Live (22-33)' if pr>=.22 else 'Longshot (<22)'
tiers=np.array([tier(x) for x in p]); tout=[]
for t in ['Elite (45%+)','Strong (33-45)','Live (22-33)','Longshot (<22)']:
    m=tiers==t
    if m.sum(): tout.append([t,int(m.sum()),round(float(y[m].mean()),3),round(float(p[m].mean()),3)])
def mrow(nm):
    yy,pp=pool(RES[nm]); return [round(brier(yy,pp),4),round(logloss(yy,pp),4),round(auc(yy,pp),3)]
bt={"seasons":[{"season":s,"n":int(len(RES['v3'][s][0])),"base":round(float(RES['v3'][s][0].mean()),3),
      "brier":round(brier(*RES['v3'][s]),4),"logloss":round(logloss(*RES['v3'][s]),4),"auc":round(auc(*RES['v3'][s]),3)} for s in TEST],
    "pooled":{"n":int(len(y)),"base":round(float(y.mean()),3),"brier":round(brier(y,p),4),
      "logloss":round(logloss(y,p),4),"auc":round(auc(y,p),3),"ece":round(ece(y,p),3)},
    "models":[["Naive: past TD rate"]+mrow('naive'),["Opportunity model"]+mrow('base'),
              ["+ carry-share + Vegas + NGS"]+mrow('v3')],
    "tiers":tout}
json.dump(bt,open('out/backtest_v3.json','w'),indent=1)
print("\nsaved out/backtest_v3.json")

# ---------------------------------------------------------------------------
# Comprehensive human-readable report (all metrics + calibration tables)
# ---------------------------------------------------------------------------
def bands(y,p,edges=(0,.05,.10,.15,.20,.25,.30,.40,.50,1.01)):
    out=[]
    for i in range(len(edges)-1):
        m=(p>=edges[i])&(p<edges[i+1])
        if m.sum(): out.append((edges[i],min(edges[i+1],1.0),int(m.sum()),float(p[m].mean()),float(y[m].mean())))
    return out
def tierrows(y,p):
    def T(x): return 'Elite (45%+)' if x>=.45 else 'Strong (33-45%)' if x>=.33 else 'Live (22-33%)' if x>=.22 else 'Longshot (<22%)'
    tr=np.array([T(x) for x in p]); out=[]
    for t in ['Elite (45%+)','Strong (33-45%)','Live (22-33%)','Longshot (<22%)']:
        m=tr==t
        if m.sum(): out.append((t,int(m.sum()),float(p[m].mean()),float(y[m].mean())))
    return out

L=[]
L.append("# EndZone Board — model performance report\n")
L.append("Anytime-touchdown model, walk-forward out-of-sample backtest.\n")
L.append("- Scoring rates fit only on seasons **before** each test year; the classifier is trained only on prior seasons; every prediction uses pre-kickoff info only.")
L.append("- Test seasons: **2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025**. Evaluation universe: active, involved skill players (RB/WR/TE/QB with ≥1 touch).")
L.append("- Features include opportunity (carry/target volume, red-zone & goal-line carry/target share), Vegas implied team total, recent-form (last-3-game) usage & trend, and **Next Gen Stats** trailing form (receiver separation & YAC-over-expected; rusher efficiency, yards-over-expected & light/stacked-box rate).")
L.append("- This report covers the **conversion** model: P(TD | the player plays). The live board multiplies it by a separate **availability** model — P(the player takes the field) — to show a true pregame probability. See `eligibility_report.md` for that two-stage backtest.")
yv,pv=pool(RES['v3']); yb,pb=pool(RES['base']); yn,pn=pool(RES['naive'])
L.append(f"- Pooled held-out sample: **{len(yv):,} player-games**, base rate **{yv.mean()*100:.1f}%**.\n")

L.append("## Pooled model comparison (2018–2025)\n")
L.append("| Model | Brier ↓ | Log loss ↓ | AUC ↑ | Calib. err (ECE) ↓ |")
L.append("|---|---|---|---|---|")
for nm,(yy,pp) in [("Naive: count past TDs",(yn,pn)),("Opportunity model",(yb,pb)),("+ carry-share + Vegas + NGS (shipped)",(yv,pv))]:
    L.append(f"| {nm} | {brier(yy,pp):.4f} | {logloss(yy,pp):.4f} | {auc(yy,pp):.3f} | {ece(yy,pp):.3f} |")

L.append("\n## Per-season (shipped model)\n")
L.append("| Season | n | Base rate | Brier | Log loss | AUC | ECE |")
L.append("|---|---|---|---|---|---|---|")
for s in TEST:
    y,p=RES['v3'][s]
    L.append(f"| {s} | {len(y):,} | {y.mean()*100:.1f}% | {brier(y,p):.4f} | {logloss(y,p):.4f} | {auc(y,p):.3f} | {ece(y,p):.3f} |")
y,p=pool(RES['v3']); L.append(f"| **Pooled** | **{len(y):,}** | **{y.mean()*100:.1f}%** | **{brier(y,p):.4f}** | **{logloss(y,p):.4f}** | **{auc(y,p):.3f}** | **{ece(y,p):.3f}** |")

L.append("\n## Per-season Brier / log loss, all models\n")
L.append("| Season | Naive Brier | Naive LogLoss | Opp Brier | Opp LogLoss | v3 Brier | v3 LogLoss |")
L.append("|---|---|---|---|---|---|---|")
for s in TEST:
    yn2,pn2=RES['naive'][s]; yb2,pb2=RES['base'][s]; yv2,pv2=RES['v3'][s]
    L.append(f"| {s} | {brier(yn2,pn2):.4f} | {logloss(yn2,pn2):.4f} | {brier(yb2,pb2):.4f} | {logloss(yb2,pb2):.4f} | {brier(yv2,pv2):.4f} | {logloss(yv2,pv2):.4f} |")

L.append("\n## Calibration table — shipped model, pooled (2018–2025)\n")
L.append("Predicted-probability band vs. the rate players in that band actually scored. Close = well calibrated.\n")
L.append("| Predicted band | n | Mean predicted | Actually scored |")
L.append("|---|---|---|---|")
for lo,hi,n,mp,ob in bands(yv,pv):
    L.append(f"| {int(lo*100)}–{int(hi*100)}% | {n:,} | {mp*100:.1f}% | {ob*100:.1f}% |")

L.append("\n## Tier hit-rate — shipped model, pooled\n")
L.append("| Tier | Players | Model avg | Actually scored |")
L.append("|---|---|---|---|")
for t,n,mp,ob in tierrows(yv,pv):
    L.append(f"| {t} | {n:,} | {mp*100:.1f}% | {ob*100:.1f}% |")

L.append("\n## Per-season calibration tables (shipped model)\n")
for s in TEST:
    y,p=RES['v3'][s]
    L.append(f"\n### {s}\n")
    L.append("| Predicted band | n | Mean predicted | Actually scored |")
    L.append("|---|---|---|---|")
    for lo,hi,n,mp,ob in bands(y,p):
        L.append(f"| {int(lo*100)}–{int(hi*100)}% | {n:,} | {mp*100:.1f}% | {ob*100:.1f}% |")

L.append("\n---\n*Metrics: Brier = mean squared error of probabilities; Log loss = negative log-likelihood; "
         "AUC = ranking (P a scorer outranks a non-scorer); ECE = mean gap between predicted and observed across deciles. Lower is better except AUC.*\n")
open('out/model_report.md','w').write("\n".join(L))
print("wrote out/model_report.md")

# ===========================================================================
# EXTENDED breakdowns: by position, by workload tier, + calibration curve SVG
# ===========================================================================
yv,pv=pool(RES['v3'])
POSp=np.concatenate([POS[s] for s in TEST]).astype(str)
VOLp=np.concatenate([VOL[s] for s in TEST])
E=[]  # extra report sections

def seg(mask):
    if mask.sum()<30: return None
    return (int(mask.sum()),brier(yv[mask],pv[mask]),logloss(yv[mask],pv[mask]),
            auc(yv[mask],pv[mask]),ece(yv[mask],pv[mask]),float(yv[mask].mean()),float(pv[mask].mean()))

E.append("\n## Performance by position (shipped model, pooled 2018–2025)\n")
E.append("| Position | n | Brier | Log loss | AUC | ECE | Model avg | Actual |")
E.append("|---|---|---|---|---|---|---|---|")
for pp in ['RB','WR','TE','QB']:
    s=seg(POSp==pp)
    if s: E.append(f"| {pp} | {s[0]:,} | {s[1]:.4f} | {s[2]:.4f} | {s[3]:.3f} | {s[4]:.3f} | {s[6]*100:.1f}% | {s[5]*100:.1f}% |")

E.append("\n## Performance by workload tier (trailing carries+targets / game)\n")
for lab,mk in [('Workhorse (≥15/g)',VOLp>=15),('Regular (8–15/g)',(VOLp>=8)&(VOLp<15)),
               ('Rotational (3–8/g)',(VOLp>=3)&(VOLp<8))]:
    s=seg(mk)
    if s:
        if 'Workload' not in "".join(E[-6:]):
            E.append("| Workload | n | Brier | Log loss | AUC | ECE | Model avg | Actual |")
            E.append("|---|---|---|---|---|---|---|---|")
        E.append(f"| {lab} | {s[0]:,} | {s[1]:.4f} | {s[2]:.4f} | {s[3]:.3f} | {s[4]:.3f} | {s[6]*100:.1f}% | {s[5]*100:.1f}% |")

# calibration curve (deciles by predicted probability)
def curve(y,p,nb=10):
    q=np.quantile(p,np.linspace(0,1,nb+1)); q[0]-=1e-9; q[-1]+=1e-9; pts=[]
    for i in range(nb):
        m=(p>q[i])&(p<=q[i+1])
        if m.sum(): pts.append((float(p[m].mean()),float(y[m].mean()),int(m.sum())))
    return pts
pts=curve(yv,pv)
E.append("\n## Calibration curve (deciles, shipped model, pooled)\n")
E.append(f"Total predictions: **{len(yv):,}**. Each row is one decile of predicted probability.\n")
E.append("| Decile | n | Mean predicted | Actually scored |")
E.append("|---|---|---|---|")
for i,(mp,ob,n) in enumerate(pts,1):
    E.append(f"| {i} | {n:,} | {mp*100:.1f}% | {ob*100:.1f}% |")
E.append("\nSee `calibration_curve.svg` for the reliability plot (predicted vs actual, with the diagonal = perfect calibration).")

# write SVG reliability plot (no deps)
W=H=340; M=44; X0=M; Y0=H-M; PW=W-2*M
def X(v): return X0+v*PW
def Y(v): return Y0-v*PW
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
svg.append(f'<rect width="{W}" height="{H}" fill="#0d1317"/>')
for g in range(0,11,2):
    v=g/10; svg.append(f'<line x1="{X(v):.0f}" y1="{Y0}" x2="{X(v):.0f}" y2="{Y0-PW}" stroke="#26333d" stroke-width="1"/>')
    svg.append(f'<line x1="{X0}" y1="{Y(v):.0f}" x2="{X0+PW}" y2="{Y(v):.0f}" stroke="#26333d" stroke-width="1"/>')
    svg.append(f'<text x="{X(v):.0f}" y="{Y0+16}" fill="#8496a2" font-size="10" text-anchor="middle">{int(v*100)}</text>')
    svg.append(f'<text x="{X0-8}" y="{Y(v)+4:.0f}" fill="#8496a2" font-size="10" text-anchor="end">{int(v*100)}</text>')
svg.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#66798a" stroke-dasharray="4 4" stroke-width="1.5"/>')
d="M "+" L ".join(f"{X(mp):.1f} {Y(ob):.1f}" for mp,ob,_ in pts)
svg.append(f'<path d="{d}" fill="none" stroke="#ff6a3d" stroke-width="2.5"/>')
for mp,ob,n in pts:
    svg.append(f'<circle cx="{X(mp):.1f}" cy="{Y(ob):.1f}" r="4" fill="#ff6a3d"/>')
svg.append(f'<text x="{W/2:.0f}" y="{H-6}" fill="#e9eff3" font-size="11" text-anchor="middle">Predicted probability (%)</text>')
svg.append(f'<text x="14" y="{H/2:.0f}" fill="#e9eff3" font-size="11" text-anchor="middle" transform="rotate(-90 14 {H/2:.0f})">Actual scoring rate (%)</text>')
svg.append(f'<text x="{X0}" y="24" fill="#e9eff3" font-size="13" font-weight="600">Calibration — anytime-TD model (2018–2025)</text>')
svg.append('</svg>')
open('out/calibration_curve.svg','w').write("".join(svg))

# append extra sections to the report
rep=open('out/model_report.md').read()
open('out/model_report.md','w').write(rep+"\n"+"\n".join(E)+"\n")
print("appended breakdowns + wrote out/calibration_curve.svg")
