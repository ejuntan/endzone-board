"""Tier-1 availability model: two-stage P(plays) x P(TD|plays) on the dense
pregame eligibility grid. Does a learned play-probability multiplier beat the
current binary 'exclude Out/Doubtful, assume everyone else plays' approach?

Universe (matches the live board): skill players, projected vol>=3/g, team
playing that week, NOT ruled Out/Doubtful/IR. Outcome = scored, with eligible
no-shows counted as 0. Stage-1 (P>=1 touch) and stage-2 (P TD|touch) both fit
on dense rows from seasons <=2021; evaluated 2022-2025.
"""
import duckdb, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import pipeline as P
con=duckdb.connect(); K=3.0
YRS=[2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]
P.build_tables(con,YRS,[2016,2017,2018,2019,2020,2021])
trail=P.team_env(con)
def c0(a): a=np.asarray(a,float); a[np.isnan(a)]=0.0; return a

INJ="read_parquet(["+",".join(f"'data/injuries_{y}.parquet'" for y in YRS)+"])"
# dense player-week grid with exact trailing (incl. last-3 recency) + NGS + injury status
con.execute(f"""create or replace table grid as
with span as (select pid,season,min(week) mn,max(week) mx, arg_max(posteam,week) tm from pg group by pid,season),
wks as (select unnest(generate_series(1,22)) wknum),
pw as (select s.pid,s.season,x.wknum,s.tm from span s join wks x on x.wknum between s.mn and s.mx),
sched as (select distinct season,week,posteam,game_id from pg),
elig as (select pw.pid,pw.season,pw.wknum,pw.tm posteam, sc.game_id
         from pw join sched sc on pw.season=sc.season and pw.wknum=sc.week and pw.tm=sc.posteam),
joined as (
  select e.pid,e.season,e.wknum,e.posteam,e.game_id,
    coalesce(g.gx_rush,0) gx_rush,coalesce(g.gx_rec,0) gx_rec,coalesce(g.carries,0) carries,coalesce(g.targets,0) targets,
    coalesce(g.n_gl,0) n_gl,coalesce(g.n_rzr,0) n_rzr,coalesce(g.n_ezt,0) n_ezt,coalesce(g.n_rzt,0) n_rzt,
    coalesce(g.scored,0) scored, gs.off_pct, (g.pid is not null)::int played
  from elig e
  left join pg g on e.pid=g.pid and e.season=g.season and e.wknum=g.week
  left join pgs gs on e.pid=gs.pid and e.season=gs.season and e.wknum=gs.week)
select *,
  coalesce(sum(gx_rush) over w,0) c_rush,coalesce(sum(gx_rec) over w,0) c_rec,
  coalesce(sum(carries) over w,0) c_car,coalesce(sum(targets) over w,0) c_tgt,
  coalesce(sum(n_gl) over w,0) c_gl,coalesce(sum(n_rzr) over w,0) c_rzr,
  coalesce(sum(n_ezt) over w,0) c_ezt,coalesce(sum(n_rzt) over w,0) c_rzt,
  coalesce(sum(scored) over w,0) c_scr, coalesce(sum(played) over w,0) c_g, avg(off_pct) over w a_snap,
  coalesce(sum(carries) over w3,0) c_car_l3,coalesce(sum(targets) over w3,0) c_tgt_l3,
  coalesce(sum(n_gl) over w3,0) c_gl_l3, coalesce(count(*) over w3,0) c_g_l3, avg(off_pct) over w3 a_snap_l3,
  coalesce(sum(played) over w3,0) p_g_l3
from joined
window w  as (partition by pid,season order by wknum rows between unbounded preceding and 1 preceding),
       w3 as (partition by pid,season order by wknum rows between 3 preceding and 1 preceding)""")

def pull(grid=True):
    src = ("grid gr", "gr", "wknum") if grid else ("pgw gr", "gr", "week")
    extra = ("gr.played, gr.c_g_l3, gr.a_snap_l3, gr.p_g_l3," if grid else
             "1 played, gr.c_g_l3, gr.a_snap_l3, gr.c_g_l3 p_g_l3,")
    return con.sql(f"""select gr.season,{src[2]} wknum,gr.posteam,po.ppos,gr.scored,
        gr.c_rush,gr.c_rec,gr.c_car,gr.c_tgt,gr.c_gl,gr.c_rzr,gr.c_ezt,gr.c_rzt,gr.c_scr,gr.c_g,gr.a_snap,
        gr.c_car_l3,gr.c_tgt_l3,gr.c_gl_l3,{extra}
        pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g,
        tr.tgl_tr,tr.trz_tr,tr.trt_tr, ve.implied,
        nt.n_sep,nt.n_yacoe,nt.n_cush,nt.n_catchp,nt.n_eff,nt.n_ryoe,nt.n_box8,nt.n_rpoe,
        np.p_sep,np.p_yacoe,np.p_cush,np.p_catchp,np.p_eff,np.p_ryoe,np.p_box8,np.p_rpoe,
        inj.status, inj.practice
      from {src[0]} left join pos po using(pid)
      left join prior pr on gr.pid=pr.pid and gr.season=pr.season
      left join teamrz_trail tr on gr.season=tr.season and {src[2]}=tr.week and gr.posteam=tr.posteam
      left join vegas ve on gr.game_id=ve.game_id and gr.posteam=ve.team
      left join ngs_trail nt on gr.pid=nt.pid and gr.season=nt.season and {src[2]}=nt.week
      left join ngs_prior np on gr.pid=np.pid and gr.season=np.season
      left join (select season,week,gsis_id pid,any_value(report_status) status,any_value(practice_status) practice
                 from {INJ} group by 1,2,3) inj on gr.pid=inj.pid and gr.season=inj.season and {src[2]}=inj.week
      """.replace("gr.game_id", "gr.game_id" if grid else "NULL")).fetchnumpy()

# global means for shrinkage + NGS fallback (from grid played rows w/ prior)
gdf=pull(grid=True)
has_pr=c0(gdf['pr_g'])>0
gm={n:float(np.mean(c0(gdf[n])[has_pr])) for n in ['pr_rush','pr_rec','pr_car','pr_tgt','pr_gl','pr_rzr','pr_ezt','pr_rzt','pr_scr']}
asn=np.asarray(gdf['a_snap'],float); gsnap=float(np.nanmean(asn[~np.isnan(asn)]))
NGS_GM=P.ngs_global_means(gdf)

def stage2_feats(d):
    hp=c0(d['pr_g'])>0; cg=c0(d['c_g'])
    season=np.asarray(d['season']);week=np.asarray(d['wknum']);team=np.asarray(d['posteam']);pos=d['ppos'].astype(str)
    def sh(cs,pp,gl): return (c0(cs)+K*np.where(hp,c0(pp),gl))/(cg+K)
    xr=sh(d['c_rush'],d['pr_rush'],gm['pr_rush']);xc=sh(d['c_rec'],d['pr_rec'],gm['pr_rec'])
    cpg=sh(d['c_car'],d['pr_car'],gm['pr_car']);tpg=sh(d['c_tgt'],d['pr_tgt'],gm['pr_tgt'])
    glp=sh(d['c_gl'],d['pr_gl'],gm['pr_gl']);rzr=sh(d['c_rzr'],d['pr_rzr'],gm['pr_rzr'])
    ezt=sh(d['c_ezt'],d['pr_ezt'],gm['pr_ezt']);rzt=sh(d['c_rzt'],d['pr_rzt'],gm['pr_rzt'])
    nv=sh(d['c_scr'],d['pr_scr'],gm['pr_scr']);aa=np.asarray(d['a_snap'],float);snap=np.where(~np.isnan(aa),aa,gsnap)
    trz=c0(d['trz_tr']);tgl=c0(d['tgl_tr']);trt=c0(d['trt_tr'])
    rzcsh=np.clip((c0(d['c_gl'])+c0(d['c_rzr']))/np.maximum(trz,1e-6),0,1.2)
    glcsh=np.clip(c0(d['c_gl'])/np.maximum(tgl,1e-6),0,1.2)
    rztsh=np.clip((c0(d['c_ezt'])+c0(d['c_rzt']))/np.maximum(trt,1e-6),0,1.2)
    imp=np.asarray(d['implied'],float);imp[np.isnan(imp)]=22.0
    te=np.array([trail.get((f"{int(season[i])}_{team[i]}",int(week[i])),2.4) for i in range(len(season))])
    g3=c0(d['c_g_l3'])
    def r3(cs,fb): v=c0(cs)/np.maximum(g3,1e-6); return np.where(g3>=1,v,fb)
    cpg3=r3(d['c_car_l3'],cpg);tpg3=r3(d['c_tgt_l3'],tpg);glp3=r3(d['c_gl_l3'],glp)
    a3=np.asarray(d['a_snap_l3'],float);snap3=np.where(~np.isnan(a3),a3,snap)
    ngs=P.ngs_cols(d,NGS_GM)
    cols={'xr':xr,'xc':xc,'xtd':xr+xc,'vol':cpg+tpg,'cpg':cpg,'tpg':tpg,'glpg':glp,'rzrpg':rzr,'eztpg':ezt,'rztpg':rzt,
          'rz_csh':rzcsh,'gl_csh':glcsh,'rz_tsh':rztsh,'exp_gl_td':glp*0.38+rzr*0.08,'implied':imp,'team_exp':te,'snap':snap,
          'naive':nv,'cg':cg,'cpg3':cpg3,'tpg3':tpg3,'glp3':glp3,'snap3':snap3,'trend_car':cpg3-cpg,'trend_gl':glp3-glp,'trend_tgt':tpg3-tpg,
          **ngs,'is_RB':(pos=='RB'),'is_WR':(pos=='WR'),'is_TE':(pos=='TE'),'is_QB':(pos=='QB')}
    return np.column_stack([cols[f] for f in P.FEATS]).astype(float), cpg+tpg

# ---- stage-1 availability features (predict >=1 touch given eligible pregame) ----
def status_code(s):  # ordinal severity from final game-status report
    m={'':0,'None':0,'Questionable':1,'Doubtful':2}; return m.get(str(s),0)
def practice_code(s):
    s=str(s)
    if 'Did Not' in s: return 2
    if 'Limited' in s: return 1
    return 0  # Full / none
def stage1_feats(d):
    hp=c0(d['pr_g'])>0; cg=c0(d['c_g']); pos=d['ppos'].astype(str)
    aa=np.asarray(d['a_snap'],float);snap=np.where(~np.isnan(aa),aa,gsnap)
    a3=np.asarray(d['a_snap_l3'],float);snap3=np.where(~np.isnan(a3),a3,snap)
    pg3=c0(d['p_g_l3'])            # games actually played in last 3 wks
    def sh(cs,pp,gl): return (c0(cs)+K*np.where(hp,c0(pp),gl))/(cg+K)
    cpg=sh(d['c_car'],d['pr_car'],gm['pr_car']);tpg=sh(d['c_tgt'],d['pr_tgt'],gm['pr_tgt'])
    stat=np.array([status_code(s) for s in d['status']],float)
    prac=np.array([practice_code(s) for s in d['practice']],float)
    cols=[stat,prac,snap,snap3,pg3,cg,cpg+tpg,hp.astype(float),
          (pos=='RB').astype(float),(pos=='WR').astype(float),(pos=='TE').astype(float),(pos=='QB').astype(float)]
    return np.column_stack(cols).astype(float)

def gbm(): return HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=60,random_state=0)
def brier(y,q):return np.mean((q-y)**2)
def ll(y,q):q=np.clip(q,1e-6,1-1e-6);return -np.mean(y*np.log(q)+(1-y)*np.log(1-q))
def auc(y,q):
    y=np.asarray(y);n1=y.sum();n0=len(y)-n1
    if n1==0 or n0==0:return float('nan')
    o=np.argsort(q,kind='mergesort');ra=np.empty(len(q),float);sp=q[o];i=0;r=1
    while i<len(sp):
        j=i
        while j+1<len(sp) and sp[j+1]==sp[i]:j+=1
        ra[o[i:j+1]]=(r+r+(j-i))/2.0;r+=(j-i+1);i=j+1
    return (ra[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def ece(y,q,b=10):
    e=np.quantile(q,np.linspace(0,1,b+1));e[0]-=1e-9;e[-1]+=1e-9;s=0.0
    for i in range(b):
        m=(q>e[i])&(q<=e[i+1])
        if m.sum():s+=m.sum()/len(q)*abs(q[m].mean()-y[m].mean())
    return s

season=np.asarray(gdf['season']);pos=gdf['ppos'].astype(str);played=c0(gdf['played']).astype(int)
scored=c0(gdf['scored']).astype(int)
X2,vol=stage2_feats(gdf); X1=stage1_feats(gdf)
skill=np.isin(pos,['RB','WR','TE','QB'])
status=np.array([str(s) if s is not None else '' for s in gdf['status']])
OUT={'Out','Doubtful','Injured Reserve'}
# stage-2 trained on PLAYED rows <=2021 (same as shipped); stage-1 on ALL eligible grid rows <=2021
tr2=skill&(played==1)&((c0(gdf['c_g'])>=1)|has_pr)&(season<=2021)
tr1=skill&(vol>=1)&((c0(gdf['c_g'])>=1)|has_pr)&(season<=2021)
m2=gbm().fit(X2[tr2],scored[tr2]); p2=m2.predict_proba(X2)[:,1]
m1=gbm().fit(X1[tr1],played[tr1]); p_play=m1.predict_proba(X1)[:,1]

# evaluation universe = the live board's: skill, vol>=3, not ruled out, has history
test=(season>=2022)&skill&(vol>=3)&((c0(gdf['c_g'])>=1)|has_pr)&(~np.isin(status,list(OUT)))
y=scored[test]
print("=== Availability two-stage — dense eligibility universe 2022-2025 ===")
print("universe n=%d  base scored rate=%.1f%%  (no-show rate=%.1f%%)"%(
    test.sum(), y.mean()*100, (1-played[test].mean())*100))
print("\nStage-1 P(plays) quality on test:  AUC=%.3f  Brier=%.4f  ECE=%.3f  mean=%.1f%% actual-played=%.1f%%"%(
    auc(played[test],p_play[test]),brier(played[test],p_play[test]),ece(played[test],p_play[test]),
    p_play[test].mean()*100, played[test].mean()*100))
print("\n%-34s %8s %9s %6s %6s"%("full-pipeline P(TD) on universe","Brier","LogLoss","AUC","ECE"))
for lab,q in [("A. current (assume eligible plays)",p2[test]),
              ("B. two-stage P(plays)xP(TD|plays)",p_play[test]*p2[test])]:
    print("%-34s %8.4f %9.4f %6.3f %6.4f"%(lab,brier(y,q),ll(y,q),auc(y,q),ece(y,q)))
# focus on the players where availability is uncertain (Questionable or DNP/Limited practice)
prac=np.array([practice_code(s) for s in gdf['practice']])
unc=test&((np.array([status_code(s) for s in gdf['status']])>=1)|(prac>=1))
yu=scored[unc]
print("\nSubset: pregame-uncertain (Questionable or Limited/DNP practice)  n=%d  no-show=%.1f%%"%(
    unc.sum(),(1-played[unc].mean())*100))
for lab,q in [("A. current",p2[unc]),("B. two-stage",p_play[unc]*p2[unc])]:
    print("  %-14s Brier=%.4f LogLoss=%.4f AUC=%.3f"%(lab,brier(yu,q),ll(yu,q),auc(yu,q)))
