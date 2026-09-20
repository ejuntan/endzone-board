"""Tier-1 test: do NGS trailing form features (separation, YAC-over-exp, RYOE,
efficiency, %8+ box) improve the raw-GBM model out-of-sample?
8-season walk-forward (2018-2025). Rates/model fit on prior seasons only.
NGS trailed per player (prior weeks this season) with prior-season fallback."""
import duckdb, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import pipeline as P
con=duckdb.connect(); TEST=list(range(2018,2026)); K=3.0
def c0(a): a=np.asarray(a,float); a[np.isnan(a)]=0.0; return a

def build_ngs(con,years):
    ys="["+",".join(str(y) for y in years)+"]"
    # weekly per-player NGS (regular+post), keyed by gsis id
    con.execute(f"""create or replace table ngsw as
      select player_gsis_id pid, season, week, avg_separation sep, avg_yac_above_expectation yacoe,
             avg_cushion cush, catch_percentage catchp
      from read_parquet('data/ngs_receiving.parquet') where week>0 and season in {ys.replace('[','(').replace(']',')')}
      union all by name
      select player_gsis_id pid, season, week, efficiency eff, rush_yards_over_expected_per_att ryoe,
             percent_attempts_gte_eight_defenders box8, rush_pct_over_expected rpoe
      from read_parquet('data/ngs_rushing.parquet') where week>0 and season in {ys.replace('[','(').replace(']',')')}""")
    # merge the two unions into one row per pid-week (rec + rush metrics)
    con.execute("""create or replace table ngm as
      select pid,season,week,
        max(sep) sep,max(yacoe) yacoe,max(cush) cush,max(catchp) catchp,
        max(eff) eff,max(ryoe) ryoe,max(box8) box8,max(rpoe) rpoe
      from (
        select player_gsis_id pid, season, week, avg_separation sep, avg_yac_above_expectation yacoe,
               avg_cushion cush, catch_percentage catchp,
               NULL::double eff, NULL::double ryoe, NULL::double box8, NULL::double rpoe
        from read_parquet('data/ngs_receiving.parquet') where week>0
        union all
        select player_gsis_id, season, week, NULL,NULL,NULL,NULL,
               efficiency, rush_yards_over_expected_per_att, percent_attempts_gte_eight_defenders, rush_pct_over_expected
        from read_parquet('data/ngs_rushing.parquet') where week>0)
      group by pid,season,week""")
    # trailing (prior weeks this season)
    con.execute("""create or replace table ngs_trail as
      select pid,season,week,
        avg(sep) over w n_sep, avg(yacoe) over w n_yacoe, avg(cush) over w n_cush, avg(catchp) over w n_catchp,
        avg(eff) over w n_eff, avg(ryoe) over w n_ryoe, avg(box8) over w n_box8, avg(rpoe) over w n_rpoe
      from ngm window w as (partition by pid,season order by week rows between unbounded preceding and 1 preceding)""")
    # prior-season per-player means (fallback)
    con.execute("""create or replace table ngs_prior as
      select pid, season+1 season,
        avg(sep) p_sep, avg(yacoe) p_yacoe, avg(cush) p_cush, avg(catchp) p_catchp,
        avg(eff) p_eff, avg(ryoe) p_ryoe, avg(box8) p_box8, avg(rpoe) p_rpoe
      from ngm group by pid,season""")

def pull(con):
    return con.sql("""select w.season,w.week,w.posteam,w.ppos,w.carries,w.targets,w.scored,
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

def build(df,trail):
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
    imp=np.asarray(df['implied'],float);imp[np.isnan(imp)]=22.0; pos=df['ppos'].astype(str)
    te=np.array([trail.get((f"{int(season[i])}_{team[i]}",int(week[i])),2.4) for i in range(len(season))])
    exp_gl_td=glp*0.38+rzr*0.08
    cur=[xr,xc,xr+xc,cpg+tpg,cpg,tpg,glp,rzr,ezt,rzt,rzcsh,glcsh,rztsh,exp_gl_td,imp,te,snap,nv,c_g,
         (pos=='RB'),(pos=='WR'),(pos=='TE'),(pos=='QB')]
    g3=c0(df['c_g_l3'])
    def r3(cs,fb): v=c0(cs)/np.maximum(g3,1e-6); return np.where(g3>=1,v,fb)
    cpg3=r3(df['c_car_l3'],cpg);tpg3=r3(df['c_tgt_l3'],tpg);glp3=r3(df['c_gl_l3'],glp)
    a3=np.asarray(df['a_snap_l3'],float); snap3=np.where(~np.isnan(a3),a3,snap)
    rec=[cpg3,tpg3,glp3,snap3,cpg3-cpg,glp3-glp,tpg3-tpg]
    # NGS: trailing this season, else prior-season, else global mean. keep a 'has NGS' flag.
    def ngs(cur_col,prior_col):
        v=np.asarray(df[cur_col],float); pv=np.asarray(df[prior_col],float)
        g=float(np.nanmean(v[~np.isnan(v)])); out=np.where(~np.isnan(v),v,np.where(~np.isnan(pv),pv,g))
        return out
    n_sep=ngs('n_sep','p_sep');n_yacoe=ngs('n_yacoe','p_yacoe');n_cush=ngs('n_cush','p_cush');n_catchp=ngs('n_catchp','p_catchp')
    n_eff=ngs('n_eff','p_eff');n_ryoe=ngs('n_ryoe','p_ryoe');n_box8=ngs('n_box8','p_box8');n_rpoe=ngs('n_rpoe','p_rpoe')
    ngs_all=[n_sep,n_yacoe,n_cush,n_catchp,n_eff,n_ryoe,n_box8,n_rpoe]
    ngs_lite=[n_sep,n_yacoe,n_ryoe,n_box8]  # most plausibly-predictive subset
    vol=cpg+tpg; sc=c0(df['scored']).astype(int); carr=c0(df['carries']);targ=c0(df['targets'])
    base=np.isin(pos,['RB','WR','TE','QB'])&((carr+targ)>=1)&((c_g>=1)|has_pr)
    return dict(Xbase=np.column_stack(cur+rec).astype(float),
                Xngs=np.column_stack(cur+rec+ngs_all).astype(float),
                Xlite=np.column_stack(cur+rec+ngs_lite).astype(float),
                sc=sc,base=base,season=season,vol=vol)

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
VAR=['base','+ngs_all','+ngs_lite']
acc={k:{'y':[],'p':[],'v':[]} for k in VAR}
for T in TEST:
    yrs=[y for y in range(2016,T+1)]
    P.build_tables(con,yrs,[y for y in range(2016,T)])
    build_ngs(con,yrs); trail=P.team_env(con); D=build(pull(con),trail)
    tr=D['base']&(D['season']<T);teM=D['base']&(D['season']==T)
    for k,X in [('base',D['Xbase']),('+ngs_all',D['Xngs']),('+ngs_lite',D['Xlite'])]:
        m=gbm().fit(X[tr],D['sc'][tr]); p=m.predict_proba(X[teM])[:,1]
        acc[k]['y'].append(D['sc'][teM]);acc[k]['p'].append(p);acc[k]['v'].append(D['vol'][teM])
    print("season",T,"done  test n=",int(teM.sum()))
print("\nper-season AUC (base -> +ngs_all):")
for i,T in enumerate(TEST):
    yb=acc['base']['y'][i];pb=acc['base']['p'][i];pn=acc['+ngs_all']['p'][i]
    print("  %d: %.4f -> %.4f  (dBrier %+.4f)"%(T,auc(yb,pb),auc(yb,pn),brier(yb,pn)-brier(yb,pb)))
print("\n8-season walk-forward (2018-2025)  n=%d"%len(np.concatenate(acc['base']['y'])))
print("%-10s | Brier  LogLoss  AUC    ECE   | WORKHORSE pred%% act%% AUC"%"variant")
for k in VAR:
    y=np.concatenate(acc[k]['y']);p=np.concatenate(acc[k]['p']);v=np.concatenate(acc[k]['v']);m=v>=15
    print("%-10s | %.4f %.4f %.4f %.4f | %4.1f %4.1f %.3f"%(
        k,brier(y,p),ll(y,p),auc(y,p),ece(y,p),p[m].mean()*100,y[m].mean()*100,auc(y[m],p[m])))
