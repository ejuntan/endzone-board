import duckdb, numpy as np, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import pipeline as P

con=duckdb.connect()
TEST=[2022,2023,2024,2025]
NEW=['rz_csh','gl_csh','rz_tsh','implied']

def features(con):
    df=con.sql("""select w.season,w.week,w.posteam,w.ppos,w.carries,w.targets,w.scored,
        w.c_rush,w.c_rec,w.c_car,w.c_tgt,w.c_gl,w.c_rzr,w.c_ezt,w.c_rzt,w.c_scr,w.c_g,w.a_snap,
        pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g,
        tr.tgl_tr,tr.trz_tr,tr.trt_tr, ve.implied
      from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season
      left join teamrz_trail tr on w.season=tr.season and w.week=tr.week and w.posteam=tr.posteam
      left join vegas ve on w.game_id=ve.game_id and w.posteam=ve.team""").fetchnumpy()
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
    cols={'xr':xr,'xc':xc,'xtd':xtd,'vol':vol,'cpg':cpg,'tpg':tpg,'glpg':glp,'rzrpg':rzr,'eztpg':ezt,'rztpg':rzt,
          'rz_csh':rz_csh,'gl_csh':gl_csh,'rz_tsh':rz_tsh,'implied':implied,'team_exp':team_exp,'snap':snap,
          'naive':nv,'cg':c_g,'is_RB':(pos=='RB'),'is_WR':(pos=='WR'),'is_TE':(pos=='TE'),'is_QB':(pos=='QB')}
    X=np.column_stack([cols[f] for f in P.FEATS]).astype(float)
    Xbase=np.column_stack([cols[f] for f in P.FEATS if f not in NEW]).astype(float)
    carries=c0(df['carries']);targets=c0(df['targets']);scored=c0(df['scored']).astype(int)
    skill=np.isin(pos,['RB','WR','TE','QB']); base=skill&((carries+targets)>=1)&((c_g>=1)|has_pr)
    return X,Xbase,scored,base,season

def fitpred(Xtr,ytr,Xte):
    clf=CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,
        l2_regularization=1.0,min_samples_leaf=60,random_state=0),method='sigmoid',cv=3)
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
for T in TEST:
    P.build_tables(con, [y for y in range(2021,T+1)], [y for y in range(2021,T)])
    trail=P.team_env(con); df=features(con); X,Xbase,y,base,season=make(df,trail)
    tr=base&(season<T); te=base&(season==T)
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
              ["+ carry-share + Vegas"]+mrow('v3')],
    "tiers":tout}
json.dump(bt,open('out/backtest_v3.json','w'),indent=1)
print("\nsaved out/backtest_v3.json")
