"""
Learned, calibrated anytime-TD model vs. the fixed-rate model, same 3-season
walk-forward backtest. For test season T: scoring rates fit on <T, gradient-
boosted classifier trained on <T (features are all trailing / pre-kickoff),
isotonic-calibrated via CV, evaluated on T.

Features per player-game (walk-forward trailing, shrunk):
  analytic xTD (rush/rec) and its Poisson prob, volume, explicit goal-line &
  end-zone/red-zone opportunity, snap share, team scoring environment,
  trailing TD rate, games of history, position one-hots.
"""
import duckdb, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
con=duckdb.connect()
def rp(s): return "read_parquet(["+",".join(f"'data/pbp_{x}.parquet'" for x in s)+"])"
ROST="read_parquet(['data/roster_2021.parquet','data/roster_2022.parquet','data/roster_2023.parquet','data/roster_2024.parquet'])"
K=3.0

def fit_rates(fs):
    F=rp(fs)
    rush={r[0]:r[1] for r in con.sql(f"""select case when yardline_100<=2 then 0 when yardline_100<=5 then 1
      when yardline_100<=10 then 2 when yardline_100<=20 then 3 when yardline_100<=50 then 4 else 5 end b,
      avg(rush_touchdown) rate from {F} where rush_attempt=1 and rusher_player_id is not null and yardline_100 is not null group by 1""").fetchall()}
    rec={(int(r[0]),int(r[1])):r[2] for r in con.sql(f"""select case when yardline_100<=5 then 0 when yardline_100<=10 then 1
      when yardline_100<=20 then 2 when yardline_100<=40 then 3 else 4 end b,
      case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end ez,
      avg(pass_touchdown) rate from {F} where pass_attempt=1 and receiver_player_id is not null and yardline_100 is not null group by 1,2""").fetchall()}
    return rush,rec
def rushc(r): return ("case when yardline_100<=2 then {0} when yardline_100<=5 then {1} when yardline_100<=10 then {2} "
    "when yardline_100<=20 then {3} when yardline_100<=50 then {4} else {5} end").format(*[r[i] for i in range(6)])
def recc(r):
    yb="case when yardline_100<=5 then 0 when yardline_100<=10 then 1 when yardline_100<=20 then 2 when yardline_100<=40 then 3 else 4 end"
    ez="(case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end)"
    return "case "+" ".join(f"when {yb}={b} and {ez}={e} then {r.get((b,e),0.0)}" for b in range(5) for e in (0,1))+" else 0 end"

FEATS=['xr','xc','xtd','vol','cpg','tpg','glpg','rzrpg','eztpg','rztpg','snap','team_exp','naive','cg','is_RB','is_WR','is_TE','is_QB']

def build(test_season):
    fs=[s for s in range(2021,test_season)]; rush,rec=fit_rates(fs)
    need=[s for s in range(2021,test_season+1)]; ALL=rp(need)
    con.execute(f"""create or replace table plays as
      select season,week,game_id,posteam,defteam,rusher_player_id pid,rusher_player_name pname,
        1 cy,0 tg,({rushc(rush)}) rrate,0.0 crate, (yardline_100<=5)::int n_gl,
        (yardline_100 between 6 and 20)::int n_rzr, 0 n_ezt,0 n_rzt, rush_touchdown td
        from {ALL} where rush_attempt=1 and rusher_player_id is not null and yardline_100 is not null
      union all
      select season,week,game_id,posteam,defteam,receiver_player_id,receiver_player_name,
        0,1,0.0,({recc(rec)}),0,0,
        (case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end),
        (case when yardline_100<=20 and not(air_yards is not null and air_yards>=yardline_100) then 1 else 0 end),
        pass_touchdown
        from {ALL} where pass_attempt=1 and receiver_player_id is not null and yardline_100 is not null""")
    con.execute("""create or replace table pg as
      select season,week,game_id,posteam,any_value(defteam) opp,pid,any_value(pname) pname,
        sum(cy) carries,sum(tg) targets,sum(rrate) gx_rush,sum(crate) gx_rec,
        sum(n_gl) n_gl,sum(n_rzr) n_rzr,sum(n_ezt) n_ezt,sum(n_rzt) n_rzt,(max(td)>0)::int scored
      from plays group by season,week,game_id,posteam,pid""")
    con.execute(f"""create or replace table pos as select gsis_id pid, any_value("position") ppos from {ROST} group by gsis_id""")
    # snap share via pfr->gsis crosswalk, per (pid, game_id)
    con.execute("""create or replace table snap as
      select pl.gsis_id pid, s.game_id, max(s.offense_pct) off_pct
      from read_parquet(['data/snaps_2021.parquet','data/snaps_2022.parquet','data/snaps_2023.parquet','data/snaps_2024.parquet']) s
      join 'data/players.parquet' pl on s.pfr_player_id = pl.pfr_id
      group by pl.gsis_id, s.game_id""")
    con.execute("""create or replace table pgs as
      select p.*, sn.off_pct from pg p left join snap sn on p.pid=sn.pid and p.game_id=sn.game_id""")
    con.execute("""create or replace table pgw as
      select p.*, po.ppos,
        coalesce(sum(gx_rush) over w,0) c_rush, coalesce(sum(gx_rec) over w,0) c_rec,
        coalesce(sum(carries) over w,0) c_car, coalesce(sum(targets) over w,0) c_tgt,
        coalesce(sum(n_gl) over w,0) c_gl, coalesce(sum(n_rzr) over w,0) c_rzr,
        coalesce(sum(n_ezt) over w,0) c_ezt, coalesce(sum(n_rzt) over w,0) c_rzt,
        coalesce(sum(scored) over w,0) c_scr, count(*) over w c_g, avg(off_pct) over w a_snap
      from pgs p left join pos po using(pid)
      window w as (partition by pid,season order by week rows between unbounded preceding and 1 preceding)""")
    con.execute("""create or replace table prior as
      select pid,season+1 season, avg(gx_rush) pr_rush,avg(gx_rec) pr_rec,avg(carries) pr_car,avg(targets) pr_tgt,
        avg(n_gl) pr_gl,avg(n_rzr) pr_rzr,avg(n_ezt) pr_ezt,avg(n_rzt) pr_rzt,avg(scored) pr_scr,
        avg(off_pct) pr_snap,count(*) pr_g from pgs group by pid,season""")
    con.execute("""create or replace table tg as
      select season,week,game_id,posteam,sum(sa) t from (
        select season,week,game_id,posteam,(max(td)>0)::int sa from plays group by season,week,game_id,posteam,pid)
        group by season,week,game_id,posteam""")
    df=con.sql("""select w.season,w.week,w.game_id,w.posteam,w.opp,w.pname,w.pid,w.ppos,w.carries,w.targets,w.scored,
        w.c_rush,w.c_rec,w.c_car,w.c_tgt,w.c_gl,w.c_rzr,w.c_ezt,w.c_rzt,w.c_scr,w.c_g,w.a_snap,
        pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g
      from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season""").fetchnumpy()
    def c(n): a=np.asarray(df[n],dtype=float); a[np.isnan(a)]=np.nan; return a
    def c0(n): a=np.asarray(df[n],dtype=float); a[np.isnan(a)]=0.0; return a
    season=np.asarray(df['season']);week=np.asarray(df['week']);game=np.asarray(df['game_id']);team=np.asarray(df['posteam'])
    pos=df['ppos'].astype(str); carries=c0('carries');targets=c0('targets');scored=c0('scored').astype(int)
    c_g=c0('c_g'); has_pr=c0('pr_g')>0
    g=lambda n: np.mean(c0(n)[has_pr])
    def shrink(cs,pp,gl): return (c0(cs)+K*np.where(has_pr,c0(pp),gl))/(c_g+K)
    xr=shrink('c_rush','pr_rush',g('pr_rush')); xc=shrink('c_rec','pr_rec',g('pr_rec'))
    vol=shrink('c_car','pr_car',g('pr_car'))+shrink('c_tgt','pr_tgt',g('pr_tgt'))
    cpg=shrink('c_car','pr_car',g('pr_car')); tpg=shrink('c_tgt','pr_tgt',g('pr_tgt'))
    glpg=shrink('c_gl','pr_gl',g('pr_gl')); rzrpg=shrink('c_rzr','pr_rzr',g('pr_rzr'))
    eztpg=shrink('c_ezt','pr_ezt',g('pr_ezt')); rztpg=shrink('c_rzt','pr_rzt',g('pr_rzt'))
    naive=shrink('c_scr','pr_scr',g('pr_scr')); xtd=xr+xc
    # snap: trailing avg, fallback prior-season then global
    a_snap=c('a_snap'); pr_snap=c('pr_snap'); gsnap=np.nanmean(a_snap[~np.isnan(a_snap)])
    snap=np.where(~np.isnan(a_snap),a_snap,np.where(~np.isnan(pr_snap),pr_snap,gsnap))
    # team env trailing
    tg=con.sql("select season,week,posteam,t from tg").fetchnumpy()
    ts=np.asarray(tg['season']);tw=np.asarray(tg['week']);tt=np.asarray(tg['posteam']);tv=np.asarray(tg['t'],float)
    tkey=np.char.add(np.char.add(ts.astype(str),'_'),tt.astype(str)); trail={}
    for k in np.unique(tkey):
        m=tkey==k;o=np.argsort(tw[m]);ww=tw[m][o];vv=tv[m][o];cs=np.cumsum(vv)
        for i in range(len(ww)): trail[(k,int(ww[i]))]=((cs[i]-vv[i])+K*2.4)/(i+K)
    team_key=np.char.add(np.char.add(season.astype(str),'_'),team.astype(str))
    team_exp=np.array([trail.get((team_key[i],int(week[i])),2.4) for i in range(len(season))])
    # fixed-rate coherent baseline
    gt=np.char.add(np.char.add(game.astype(str),'|'),team.astype(str)); p_fix=np.zeros(len(season))
    for k in np.unique(gt):
        m=np.where(gt==k)[0]; s=xtd[m].sum()
        if s>0: p_fix[m]=1-np.exp(-team_exp[m]*(xtd[m]/s))
    X=np.column_stack([xr,xc,xtd,vol,cpg,tpg,glpg,rzrpg,eztpg,rztpg,snap,team_exp,naive,c_g,
                       (pos=='RB'),(pos=='WR'),(pos=='TE'),(pos=='QB')]).astype(float)
    skill=np.isin(pos,['RB','WR','TE','QB']); base=skill&((carries+targets)>=1)&((c_g>=1)|has_pr)
    return dict(season=season,week=week,pname=np.asarray(df['pname']).astype(str),
        team=team,opp=np.asarray(df['opp']).astype(str),pos=pos,
        X=X,y=scored,p_fix=p_fix,p_naive=naive,base=base,
        xr=xr,xc=xc,xtd=xtd,cpg=cpg,tpg=tpg,team_exp=team_exp,snap=snap)

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

TEST=[2022,2023,2024]
D={s:build(s) for s in TEST}
res_fix={}; res_gbm={}
for T in TEST:
    d=D[T]
    tr=(d['season']<T)&d['base']; te=(d['season']==T)&d['base']
    clf=CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,
            l2_regularization=1.0,min_samples_leaf=60,random_state=0),
        method='sigmoid',cv=3)
    clf.fit(d['X'][tr],d['y'][tr])
    p=clf.predict_proba(d['X'][te])[:,1]
    res_gbm[T]=(d['y'][te],p); res_fix[T]=(d['y'][te],d['p_fix'][te])

def report(name,res):
    y=np.concatenate([res[s][0] for s in TEST]); p=np.concatenate([res[s][1] for s in TEST])
    print(f"\n{name}"); print(f"{'season':<8}{'Brier':>9}{'LogLoss':>9}{'AUC':>7}")
    for s in TEST:
        yy,pp=res[s]; print(f"{s:<8}{brier(yy,pp):>9.4f}{logloss(yy,pp):>9.4f}{auc(yy,pp):>7.3f}")
    print(f"{'POOLED':<8}{brier(y,p):>9.4f}{logloss(y,p):>9.4f}{auc(y,p):>7.3f}")
    return brier(y,p),logloss(y,p),auc(y,p)

print("="*60)
bf=report("FIXED-RATE model (current)",res_fix)
bg=report("LEARNED GBM + snaps (new)",res_gbm)
print("\nDelta (GBM - fixed):  Brier %+.4f   LogLoss %+.4f   AUC %+.3f"%(bg[0]-bf[0],bg[1]-bf[1],bg[2]-bf[2]))

# ---------------------------------------------------------------------------
# EXPORT for web board: GBM projections for 2024 wk15 + refreshed backtest
# ---------------------------------------------------------------------------
import json
# pooled arrays
def pool(res): 
    y=np.concatenate([res[s][0] for s in TEST]); p=np.concatenate([res[s][1] for s in TEST]); return y,p
res_naive={s:(D[s]['y'][(D[s]['season']==s)&D[s]['base']], D[s]['p_naive'][(D[s]['season']==s)&D[s]['base']]) for s in TEST}
yG,pG=pool(res_gbm)
def ece(y,p,b=10):
    e=np.quantile(p,np.linspace(0,1,b+1));e[0]-=1e-9;e[-1]+=1e-9;s=0.0
    for i in range(b):
        m=(p>e[i])&(p<=e[i+1])
        if m.sum():s+=m.sum()/len(p)*abs(p[m].mean()-y[m].mean())
    return s
def tier(pr): return 'Elite (45%+)' if pr>=.45 else 'Strong (33-45)' if pr>=.33 else 'Live (22-33)' if pr>=.22 else 'Longshot (<22)'
tiers=np.array([tier(x) for x in pG]); tiers_out=[]
for t in ['Elite (45%+)','Strong (33-45)','Live (22-33)','Longshot (<22)']:
    m=tiers==t
    if m.sum(): tiers_out.append([t,int(m.sum()),round(float(yG[m].mean()),3),round(float(pG[m].mean()),3)])
def mrow(res):
    y,p=pool(res); return [round(brier(y,p),4),round(logloss(y,p),4),round(auc(y,p),3)]
bt={
 "seasons":[{"season":s,"n":int(len(res_gbm[s][0])),"base":round(float(res_gbm[s][0].mean()),3),
   "brier":round(brier(*res_gbm[s]),4),"logloss":round(logloss(*res_gbm[s]),4),"auc":round(auc(*res_gbm[s]),3)} for s in TEST],
 "pooled":{"n":int(len(yG)),"base":round(float(yG.mean()),3),"brier":round(brier(yG,pG),4),
   "logloss":round(logloss(yG,pG),4),"auc":round(auc(yG,pG),3),"ece":round(ece(yG,pG),3)},
 "models":[["Naive: count past TDs"]+mrow(res_naive),
           ["Fixed-rate opportunity"]+mrow(res_fix),
           ["Learned GBM + snaps"]+mrow(res_gbm)],
 "tiers":tiers_out}
rush,rec=fit_rates([2021,2022,2023])
json.dump(bt,open('out/backtest_gbm.json','w'),indent=1)

# projections: retrain on <2024, predict 2024 wk15
d=D[2024]; tr=(d['season']<2024)&d['base']
clf=CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,
    l2_regularization=1.0,min_samples_leaf=60,random_state=0),method='sigmoid',cv=3)
clf.fit(d['X'][tr],d['y'][tr])
sel=np.where((d['season']==2024)&(d['week']==15)&d['base'])[0]
pw=clf.predict_proba(d['X'][sel])[:,1]
players=[]
for i,idx in enumerate(sel):
    players.append({"name":d['pname'][idx],"team":str(d['team'][idx]),"opp":d['opp'][idx],"pos":str(d['pos'][idx]),
        "cpg":round(float(d['cpg'][idx]),1),"tpg":round(float(d['tpg'][idx]),1),
        "xrush":round(float(d['xr'][idx]),3),"xrec":round(float(d['xc'][idx]),3),"xtd":round(float(d['xtd'][idx]),3),
        "snap":round(float(d['snap'][idx])*100,0),"teamExp":round(float(d['team_exp'][idx]),2),
        "chance":round(float(pw[i]),4)})
players.sort(key=lambda x:-x['chance']); players=players[:80]
meta={"season":2024,"week":15,
      "rush_rates":{str(k):round(v,3) for k,v in sorted(rush.items())},
      "rec_rates":{f'{b}_{e}':round(rec.get((b,e),0),3) for b in range(5) for e in (0,1)},
      "bt":bt}
json.dump({"meta":meta,"players":players},open('out/board_data_gbm.json','w'),indent=1)
print("\nExported board_data_gbm.json (%d players) and backtest_gbm.json"%len(players))
print("Top6:",[(p['name'],p['chance']) for p in players[:6]])
