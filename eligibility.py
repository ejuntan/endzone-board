"""Clean pregame player-eligibility backtest.
Instead of conditioning on a player getting >=1 touch (game-day info), we predict
for every player who was ELIGIBLE pregame: within their active span, on a week
their team plays, not ruled Out/Doubtful/IR, with projected volume >=3/g.
Players who were eligible but did not play/score count as 0s. Features are the
pregame trailing snapshot (dense grid, so trailing is exact through week-1).
Train raw GBM on played rows <=2021; test eligibility universe 2022-2025."""
import duckdb, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import pipeline as P
con=duckdb.connect(); K=3.0
P.build_tables(con,[2016,2017,2018,2019,2020,2021,2022,2023,2024,2025],[2016,2017,2018,2019,2020,2021])
trail=P.team_env(con)
def c0(a): a=np.asarray(a,float); a[np.isnan(a)]=0.0; return a

# dense player-week grid with exact trailing + eligibility + outcome
con.execute("""create or replace table grid as
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
  coalesce(sum(scored) over w,0) c_scr, coalesce(sum(played) over w,0) c_g, avg(off_pct) over w a_snap
from joined window w as (partition by pid,season order by wknum rows between unbounded preceding and 1 preceding)""")

df=con.sql("""select gr.season,gr.wknum,gr.posteam,po.ppos,gr.scored,gr.played,
    gr.c_rush,gr.c_rec,gr.c_car,gr.c_tgt,gr.c_gl,gr.c_rzr,gr.c_ezt,gr.c_rzt,gr.c_scr,gr.c_g,gr.a_snap,
    pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g,
    tr.tgl_tr,tr.trz_tr,tr.trt_tr, ve.implied, inj.status
  from grid gr left join pos po using(pid)
  left join prior pr on gr.pid=pr.pid and gr.season=pr.season
  left join teamrz_trail tr on gr.season=tr.season and gr.wknum=tr.week and gr.posteam=tr.posteam
  left join vegas ve on gr.game_id=ve.game_id and gr.posteam=ve.team
  left join (select season,week,gsis_id pid,any_value(report_status) status
             from read_parquet(['data/injuries_2022.parquet','data/injuries_2023.parquet','data/injuries_2024.parquet','data/injuries_2025.parquet'])
             group by 1,2,3) inj on gr.pid=inj.pid and gr.season=inj.season and gr.wknum=inj.week
""").fetchnumpy()

has_pr=c0(df['pr_g'])>0; c_g=c0(df['c_g']); season=np.asarray(df['season']);week=np.asarray(df['wknum']);team=np.asarray(df['posteam'])
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
cols=[xr,xc,xr+xc,cpg+tpg,cpg,tpg,glp,rzr,ezt,rzt,rzcsh,glcsh,rztsh,glp*0.38+rzr*0.08,imp,te,snap,nv,c_g,
      (pos=='RB'),(pos=='WR'),(pos=='TE'),(pos=='QB')]
X=np.column_stack(cols).astype(float); sc=c0(df['scored']).astype(int); vol=cpg+tpg
status=np.array([str(s) if s is not None else '' for s in df['status']])
OUT={'Out','Doubtful','Injured Reserve'}
skill=np.isin(pos,['RB','WR','TE','QB'])
# TRAIN on the same involved rows the shipped model uses, seasons <=2021
tdf=con.sql("""select w.season,w.ppos,w.carries,w.targets,w.scored,w.c_rush,w.c_rec,w.c_car,w.c_tgt,
    w.c_gl,w.c_rzr,w.c_ezt,w.c_rzt,w.c_scr,w.c_g,w.a_snap,w.posteam,w.week,
    pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g,
    tr.tgl_tr,tr.trz_tr,tr.trt_tr, ve.implied
  from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season
  left join teamrz_trail tr on w.season=tr.season and w.week=tr.week and w.posteam=tr.posteam
  left join vegas ve on w.game_id=ve.game_id and w.posteam=ve.team""").fetchnumpy()
def feat(d):
    hp=c0(d['pr_g'])>0; cg=c0(d['c_g'])
    def s2(cs,pp,gl): return (c0(cs)+K*np.where(hp,c0(pp),gl))/(cg+K)
    xr=s2(d['c_rush'],d['pr_rush'],gm['pr_rush']);xc=s2(d['c_rec'],d['pr_rec'],gm['pr_rec'])
    cpg=s2(d['c_car'],d['pr_car'],gm['pr_car']);tpg=s2(d['c_tgt'],d['pr_tgt'],gm['pr_tgt'])
    glp=s2(d['c_gl'],d['pr_gl'],gm['pr_gl']);rzr=s2(d['c_rzr'],d['pr_rzr'],gm['pr_rzr'])
    ezt=s2(d['c_ezt'],d['pr_ezt'],gm['pr_ezt']);rzt=s2(d['c_rzt'],d['pr_rzt'],gm['pr_rzt'])
    nv=s2(d['c_scr'],d['pr_scr'],gm['pr_scr']);aa=np.asarray(d['a_snap'],float);sn=np.where(~np.isnan(aa),aa,gsnap)
    tz=c0(d['trz_tr']);tg=c0(d['tgl_tr']);tt=c0(d['trt_tr'])
    rc=np.clip((c0(d['c_gl'])+c0(d['c_rzr']))/np.maximum(tz,1e-6),0,1.2)
    gc=np.clip(c0(d['c_gl'])/np.maximum(tg,1e-6),0,1.2)
    rt=np.clip((c0(d['c_ezt'])+c0(d['c_rzt']))/np.maximum(tt,1e-6),0,1.2)
    im=np.asarray(d['implied'],float);im[np.isnan(im)]=22.0; ps=d['ppos'].astype(str)
    sea=np.asarray(d['season']);wk=np.asarray(d['week']);tm=np.asarray(d['posteam'])
    tex=np.array([trail.get((f"{int(sea[i])}_{tm[i]}",int(wk[i])),2.4) for i in range(len(sea))])
    return np.column_stack([xr,xc,xr+xc,cpg+tpg,cpg,tpg,glp,rzr,ezt,rzt,rc,gc,rt,glp*0.38+rzr*0.08,im,tex,sn,nv,cg,
        (ps=='RB'),(ps=='WR'),(ps=='TE'),(ps=='QB')]).astype(float), c0(d['scored']).astype(int)
Xtr,ytr=feat(tdf); tseas=np.asarray(tdf['season']); tcar=c0(tdf['carries']);ttar=c0(tdf['targets']); thp=c0(tdf['pr_g'])>0; tcg=c0(tdf['c_g'])
tskill=np.isin(tdf['ppos'].astype(str),['RB','WR','TE','QB']); tmask=tskill&((tcar+ttar)>=1)&((tcg>=1)|thp)&(tseas<=2021)
clf=HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,l2_regularization=1.0,min_samples_leaf=60,random_state=0)
clf.fit(Xtr[tmask],ytr[tmask])
p=clf.predict_proba(X)[:,1]

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

test=(season>=2022)&skill&(vol>=3)&(c_g>=1)
played=c0(df['played']).astype(int)
elig=test&(~np.isin(status,list(OUT)))
def rep(label,mask):
    y=sc[mask];q=p[mask]
    print(f"{label:<40} n={mask.sum():>6,}  base={y.mean()*100:4.1f}%  Brier={brier(y,q):.4f}  LogLoss={ll(y,q):.4f}  AUC={auc(y,q):.3f}  ECE={ece(y,q):.3f}")
print("=== Clean pregame player-eligibility backtest (2022-2025) ===")
print("Universe: skill players, projected vol>=3/g, team playing that week, not ruled Out/Doubtful/IR.")
print("Outcome counts benched/scratched eligible players as 0 (no touch-conditioning).\n")
rep("Eligible (pregame) — ALL", elig)
rep("  of which actually played", elig&(played==1))
rep("  of which eligible no-show (0s)", elig&(played==0))
noshow=(elig&(played==0)).sum(); print(f"\nEligible-but-did-not-play rate: {noshow/max(1,elig.sum())*100:.1f}%  ({noshow:,} of {elig.sum():,})")
print("\nFor contrast, the touch-conditioned universe (current report) restricts to played rows only.")
