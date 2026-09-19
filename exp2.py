"""Raw GBM (no calibration wrapper): current features vs + inside-10 + team RZ TD rate.
Focus: workhorse tier (trailing carries+targets >= 15/g), walk-forward 2022-2025."""
import duckdb, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import pipeline as P
con=duckdb.connect(); TEST=[2022,2023,2024,2025]; K=3.0
def c0(a): a=np.asarray(a,float); a[np.isnan(a)]=0.0; return a
def pull(con):
    return con.sql("""select w.season,w.week,w.posteam,w.ppos,w.carries,w.targets,w.scored,
      w.c_rush,w.c_rec,w.c_car,w.c_tgt,w.c_gl,w.c_rzr,w.c_ezt,w.c_rzt,w.c_i10,w.c_scr,w.c_g,w.a_snap,
      pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_i10,pr.pr_scr,pr.pr_snap,pr.pr_g,
      tr.tgl_tr,tr.trz_tr,tr.trt_tr,tr.rzpl_tr,tr.rztd_tr, ve.implied
      from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season
      left join teamrz_trail tr on w.season=tr.season and w.week=tr.week and w.posteam=tr.posteam
      left join vegas ve on w.game_id=ve.game_id and w.posteam=ve.team""").fetchnumpy()
def build(df,trail):
    has_pr=c0(df['pr_g'])>0; c_g=c0(df['c_g']); season=np.asarray(df['season']);week=np.asarray(df['week']);team=np.asarray(df['posteam'])
    gm={n:float(np.mean(c0(df[n])[has_pr])) for n in ['pr_rush','pr_rec','pr_car','pr_tgt','pr_gl','pr_rzr','pr_ezt','pr_rzt','pr_i10','pr_scr']}
    a=np.asarray(df['a_snap'],float); gsnap=float(np.nanmean(a[~np.isnan(a)]))
    def sh(cs,pp,gl): return (c0(cs)+K*np.where(has_pr,c0(pp),gl))/(c_g+K)
    xr=sh(df['c_rush'],df['pr_rush'],gm['pr_rush']);xc=sh(df['c_rec'],df['pr_rec'],gm['pr_rec'])
    cpg=sh(df['c_car'],df['pr_car'],gm['pr_car']);tpg=sh(df['c_tgt'],df['pr_tgt'],gm['pr_tgt'])
    glp=sh(df['c_gl'],df['pr_gl'],gm['pr_gl']);rzr=sh(df['c_rzr'],df['pr_rzr'],gm['pr_rzr'])
    ezt=sh(df['c_ezt'],df['pr_ezt'],gm['pr_ezt']);rzt=sh(df['c_rzt'],df['pr_rzt'],gm['pr_rzt'])
    i10=sh(df['c_i10'],df['pr_i10'],gm['pr_i10'])
    nv=sh(df['c_scr'],df['pr_scr'],gm['pr_scr']);snap=np.where(~np.isnan(a),a,gsnap)
    trz=c0(df['trz_tr']);tgl=c0(df['tgl_tr']);trt=c0(df['trt_tr']);rzpl=c0(df['rzpl_tr']);rztd=c0(df['rztd_tr'])
    rzcsh=np.clip((c0(df['c_gl'])+c0(df['c_rzr']))/np.maximum(trz,1e-6),0,1.2)
    glcsh=np.clip(c0(df['c_gl'])/np.maximum(tgl,1e-6),0,1.2)
    rztsh=np.clip((c0(df['c_ezt'])+c0(df['c_rzt']))/np.maximum(trt,1e-6),0,1.2)
    team_rz_td=np.where(rzpl>=10, rztd/np.maximum(rzpl,1e-6), 0.205)
    imp=np.asarray(df['implied'],float);imp[np.isnan(imp)]=22.0
    pos=df['ppos'].astype(str)
    te=np.array([trail.get((f"{int(season[i])}_{team[i]}",int(week[i])),2.4) for i in range(len(season))])
    cur=[xr,xc,xr+xc,cpg+tpg,cpg,tpg,glp,rzr,ezt,rzt,rzcsh,glcsh,rztsh,imp,te,snap,nv,c_g,
         (pos=='RB'),(pos=='WR'),(pos=='TE'),(pos=='QB')]
    Xcur=np.column_stack(cur).astype(float)
    Xnew=np.column_stack(cur[:13]+[i10,team_rz_td]+cur[13:]).astype(float)
    vol=cpg+tpg; sc=c0(df['scored']).astype(int); carr=c0(df['carries']);targ=c0(df['targets'])
    base=np.isin(pos,['RB','WR','TE','QB'])&((carr+targ)>=1)&((c_g>=1)|has_pr)
    return Xcur,Xnew,sc,base,season,vol
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
acc={'raw_cur':{'y':[],'p':[],'v':[]},'raw_new':{'y':[],'p':[],'v':[]}}
for T in TEST:
    P.build_tables(con,[y for y in range(2021,T+1)],[y for y in range(2021,T)])
    trail=P.team_env(con); df=pull(con); Xc,Xn,sc,base,season,vol=build(df,trail)
    tr=base&(season<T);teM=base&(season==T)
    for k,X in [('raw_cur',Xc),('raw_new',Xn)]:
        m=gbm().fit(X[tr],sc[tr]); p=m.predict_proba(X[teM])[:,1]
        acc[k]['y'].append(sc[teM]);acc[k]['p'].append(p);acc[k]['v'].append(vol[teM])
    print("season",T,"done")
print("\n%-9s | Brier LogLoss AUC ECE | WORKHORSE n pred%% act%% ECE AUC"%"variant")
for k in acc:
    y=np.concatenate(acc[k]['y']);p=np.concatenate(acc[k]['p']);v=np.concatenate(acc[k]['v']);m=v>=15
    print("%-9s | %.4f %.4f %.3f %.3f | %4d %4.1f %4.1f %.3f %.3f"%(
        k,brier(y,p),ll(y,p),auc(y,p),ece(y,p),m.sum(),p[m].mean()*100,y[m].mean()*100,ece(y[m],p[m]),auc(y[m],p[m])))
