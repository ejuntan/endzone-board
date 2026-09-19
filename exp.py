"""Diagnose the high-workload under-prediction: test calibration method and
shrinkage strength (K) on the CURRENT features, walk-forward 2022-2025.
Focus metric: workhorse tier (trailing carries+targets >= 15/game)."""
import duckdb, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import pipeline as P
con=duckdb.connect(); TEST=[2022,2023,2024,2025]
def c0(a): a=np.asarray(a,float); a[np.isnan(a)]=0.0; return a
def pull(con):
    return con.sql("""select w.season,w.week,w.posteam,w.ppos,w.carries,w.targets,w.scored,
      w.c_rush,w.c_rec,w.c_car,w.c_tgt,w.c_gl,w.c_rzr,w.c_ezt,w.c_rzt,w.c_scr,w.c_g,w.a_snap,
      pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g,
      tr.tgl_tr,tr.trz_tr,tr.trt_tr, ve.implied
      from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season
      left join teamrz_trail tr on w.season=tr.season and w.week=tr.week and w.posteam=tr.posteam
      left join vegas ve on w.game_id=ve.game_id and w.posteam=ve.team""").fetchnumpy()
def feats(df,trail,K):
    has_pr=c0(df['pr_g'])>0; c_g=c0(df['c_g']); season=np.asarray(df['season']);week=np.asarray(df['week']);team=np.asarray(df['posteam'])
    gm={n:float(np.mean(c0(df[n])[has_pr])) for n in ['pr_rush','pr_rec','pr_car','pr_tgt','pr_gl','pr_rzr','pr_ezt','pr_rzt','pr_scr']}
    a=np.asarray(df['a_snap'],float); gsnap=float(np.nanmean(a[~np.isnan(a)]))
    def sh(cs,pp,gl): return (c0(cs)+K*np.where(has_pr,c0(pp),gl))/(c_g+K)
    xr=sh(df['c_rush'],df['pr_rush'],gm['pr_rush']);xc=sh(df['c_rec'],df['pr_rec'],gm['pr_rec'])
    cpg=sh(df['c_car'],df['pr_car'],gm['pr_car']);tpg=sh(df['c_tgt'],df['pr_tgt'],gm['pr_tgt'])
    glp=sh(df['c_gl'],df['pr_gl'],gm['pr_gl']);rzr=sh(df['c_rzr'],df['pr_rzr'],gm['pr_rzr'])
    ezt=sh(df['c_ezt'],df['pr_ezt'],gm['pr_ezt']);rzt=sh(df['c_rzt'],df['pr_rzt'],gm['pr_rzt'])
    nv=sh(df['c_scr'],df['pr_scr'],gm['pr_scr']);snap=np.where(~np.isnan(a),a,gsnap)
    trz=c0(df['trz_tr']);tgl=c0(df['tgl_tr']);trt=c0(df['trt_tr'])
    rzcsh=np.clip((c0(df['c_gl'])+c0(df['c_rzr']))/np.maximum(trz,1e-6),0,1.2)
    glcsh=np.clip(c0(df['c_gl'])/np.maximum(tgl,1e-6),0,1.2)
    rztsh=np.clip((c0(df['c_ezt'])+c0(df['c_rzt']))/np.maximum(trt,1e-6),0,1.2)
    imp=np.asarray(df['implied'],float);imp[np.isnan(imp)]=22.0
    pos=df['ppos'].astype(str)
    te=np.array([trail.get((f"{int(season[i])}_{team[i]}",int(week[i])),2.4) for i in range(len(season))])
    X=np.column_stack([xr,xc,xr+xc,cpg+tpg,cpg,tpg,glp,rzr,ezt,rzt,rzcsh,glcsh,rztsh,imp,te,snap,nv,c_g,
                       (pos=='RB'),(pos=='WR'),(pos=='TE'),(pos=='QB')]).astype(float)
    vol=cpg+tpg; sc=c0(df['scored']).astype(int); carr=c0(df['carries']);targ=c0(df['targets'])
    base=np.isin(pos,['RB','WR','TE','QB'])&((carr+targ)>=1)&((c_g>=1)|has_pr)
    return X,sc,base,season,vol
def gbm(): return HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=60,random_state=0)
def brier(y,p):return np.mean((p-y)**2)
def ll(y,p):p=np.clip(p,1e-6,1-1e-6);return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
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

variants={'sigmoid_K3':('sigmoid',3.0),'isotonic_K3':('isotonic',3.0),'raw_K3':('none',3.0),
          'sigmoid_K1.5':('sigmoid',1.5),'isotonic_K1.5':('isotonic',1.5)}
acc={k:{'y':[],'p':[],'v':[]} for k in variants}
for T in TEST:
    P.build_tables(con,[y for y in range(2021,T+1)],[y for y in range(2021,T)])
    trail=P.team_env(con); df=pull(con)
    for k,(cal,K) in variants.items():
        X,sc,base,season,vol=feats(df,trail,K)
        tr=base&(season<T);teM=base&(season==T)
        if cal=='none': m=gbm().fit(X[tr],sc[tr]); p=m.predict_proba(X[teM])[:,1]
        else: m=CalibratedClassifierCV(gbm(),method=cal,cv=3).fit(X[tr],sc[tr]); p=m.predict_proba(X[teM])[:,1]
        acc[k]['y'].append(sc[teM]); acc[k]['p'].append(p); acc[k]['v'].append(vol[teM])
    print("season",T,"done")

print("\n%-14s | overall: Brier LogLoss AUC ECE | WORKHORSE(>=15/g): n pred%% act%% ECE"%"variant")
for k in variants:
    y=np.concatenate(acc[k]['y']);p=np.concatenate(acc[k]['p']);v=np.concatenate(acc[k]['v'])
    m=v>=15
    print("%-14s |          %.4f  %.4f %.3f %.3f | %5d  %4.1f  %4.1f  %.3f"%(
        k,brier(y,p),ll(y,p),auc(y,p),ece(y,p),m.sum(),p[m].mean()*100,y[m].mean()*100,ece(y[m],p[m])))
