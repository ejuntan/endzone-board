"""
Project the UPCOMING week (2026 Week 2) with the learned model.
Rates fit on 2021-2025; GBM trained on all completed player-games through
2026 Week 1; each player's features are their current trailing state entering
the week (2026-to-date blended with 2025 via shrinkage). Opponents from the
real schedule. Nothing here uses any information from the unplayed games.
"""
import duckdb, numpy as np, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
con=duckdb.connect()
SEASON, UPCOMING = 2026, 2
YEARS=[2021,2022,2023,2024,2025,2026]
def rp(s): return "read_parquet(["+",".join(f"'data/pbp_{x}.parquet'" for x in s)+"])"
ALL=rp(YEARS)
ROST="read_parquet(["+",".join(f"'data/roster_{y}.parquet'" for y in YEARS)+"])"
SNAPS="read_parquet(["+",".join(f"'data/snaps_{y}.parquet'" for y in [2021,2022,2023,2024,2025,2026])+"])"
K=3.0
FEATS=['xr','xc','xtd','vol','cpg','tpg','glpg','rzrpg','eztpg','rztpg','snap','team_exp','naive','cg','is_RB','is_WR','is_TE','is_QB']

rush={r[0]:r[1] for r in con.sql(f"""select case when yardline_100<=2 then 0 when yardline_100<=5 then 1
  when yardline_100<=10 then 2 when yardline_100<=20 then 3 when yardline_100<=50 then 4 else 5 end b,
  avg(rush_touchdown) rate from {rp([2021,2022,2023,2024,2025])} where rush_attempt=1 and rusher_player_id is not null and yardline_100 is not null group by 1""").fetchall()}
rec={(int(r[0]),int(r[1])):r[2] for r in con.sql(f"""select case when yardline_100<=5 then 0 when yardline_100<=10 then 1
  when yardline_100<=20 then 2 when yardline_100<=40 then 3 else 4 end b,
  case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end ez,
  avg(pass_touchdown) rate from {rp([2021,2022,2023,2024,2025])} where pass_attempt=1 and receiver_player_id is not null and yardline_100 is not null group by 1,2""").fetchall()}
rushc=("case when yardline_100<=2 then {0} when yardline_100<=5 then {1} when yardline_100<=10 then {2} "
  "when yardline_100<=20 then {3} when yardline_100<=50 then {4} else {5} end").format(*[rush[i] for i in range(6)])
_yb="case when yardline_100<=5 then 0 when yardline_100<=10 then 1 when yardline_100<=20 then 2 when yardline_100<=40 then 3 else 4 end"
_ez="(case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end)"
recc="case "+" ".join(f"when {_yb}={b} and {_ez}={e} then {rec.get((b,e),0.0)}" for b in range(5) for e in (0,1))+" else 0 end"

con.execute(f"""create or replace table plays as
  select season,week,game_id,posteam,defteam,rusher_player_id pid,rusher_player_name pname,
    1 cy,0 tg,({rushc}) rrate,0.0 crate,(yardline_100<=5)::int n_gl,(yardline_100 between 6 and 20)::int n_rzr,
    0 n_ezt,0 n_rzt,rush_touchdown td
    from {ALL} where rush_attempt=1 and rusher_player_id is not null and yardline_100 is not null
  union all
  select season,week,game_id,posteam,defteam,receiver_player_id,receiver_player_name,0,1,0.0,({recc}),0,0,
    ({_ez}),(case when yardline_100<=20 and not({_ez}=1) then 1 else 0 end),pass_touchdown
    from {ALL} where pass_attempt=1 and receiver_player_id is not null and yardline_100 is not null""")
con.execute("""create or replace table pg as
  select season,week,game_id,posteam,pid,any_value(pname) pname,sum(cy) carries,sum(tg) targets,
    sum(rrate) gx_rush,sum(crate) gx_rec,sum(n_gl) n_gl,sum(n_rzr) n_rzr,sum(n_ezt) n_ezt,sum(n_rzt) n_rzt,
    (max(td)>0)::int scored from plays group by season,week,game_id,posteam,pid""")
con.execute(f"""create or replace table pos as select gsis_id pid, any_value("position") ppos from {ROST} group by gsis_id""")
con.execute(f"""create or replace table snap as select pl.gsis_id pid, s.game_id, max(s.offense_pct) off_pct
  from {SNAPS} s join 'data/players.parquet' pl on s.pfr_player_id=pl.pfr_id group by pl.gsis_id, s.game_id""")
con.execute("""create or replace table pgs as select p.*, sn.off_pct from pg p left join snap sn on p.pid=sn.pid and p.game_id=sn.game_id""")
con.execute("""create or replace table pgw as
  select p.*, po.ppos,
    coalesce(sum(gx_rush) over w,0) c_rush,coalesce(sum(gx_rec) over w,0) c_rec,
    coalesce(sum(carries) over w,0) c_car,coalesce(sum(targets) over w,0) c_tgt,
    coalesce(sum(n_gl) over w,0) c_gl,coalesce(sum(n_rzr) over w,0) c_rzr,
    coalesce(sum(n_ezt) over w,0) c_ezt,coalesce(sum(n_rzt) over w,0) c_rzt,
    coalesce(sum(scored) over w,0) c_scr,count(*) over w c_g, avg(off_pct) over w a_snap
  from pgs p left join pos po using(pid)
  window w as (partition by pid,season order by week rows between unbounded preceding and 1 preceding)""")
con.execute("""create or replace table prior as
  select pid,season+1 season,avg(gx_rush) pr_rush,avg(gx_rec) pr_rec,avg(carries) pr_car,avg(targets) pr_tgt,
    avg(n_gl) pr_gl,avg(n_rzr) pr_rzr,avg(n_ezt) pr_ezt,avg(n_rzt) pr_rzt,avg(scored) pr_scr,avg(off_pct) pr_snap,
    count(*) pr_g from pgs group by pid,season""")
con.execute("""create or replace table tg as select season,week,posteam,sum(sa) t from (
    select season,week,game_id,posteam,(max(td)>0)::int sa from plays group by season,week,game_id,posteam,pid)
    group by season,week,posteam""")

# ---- TRAINING rows (all completed player-games) ----
df=con.sql("""select w.season,w.week,w.ppos,w.carries,w.targets,w.scored,
    w.c_rush,w.c_rec,w.c_car,w.c_tgt,w.c_gl,w.c_rzr,w.c_ezt,w.c_rzt,w.c_scr,w.c_g,w.a_snap,
    pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g
  from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season""").fetchnumpy()
def c0(a): a=np.asarray(a,float); a[np.isnan(a)]=0.0; return a
season=np.asarray(df['season']);week=np.asarray(df['week']);pos=df['ppos'].astype(str)
carries=c0(df['carries']);targets=c0(df['targets']);scored=c0(df['scored']).astype(int)
c_g=c0(df['c_g']); has_pr=c0(df['pr_g'])>0
gm={}
for nm in ['pr_rush','pr_rec','pr_car','pr_tgt','pr_gl','pr_rzr','pr_ezt','pr_rzt','pr_scr']:
    gm[nm]=float(np.mean(c0(df[nm])[has_pr]))
a_snap=np.asarray(df['a_snap'],float); pr_snap=np.asarray(df['pr_snap'],float)
gsnap=float(np.nanmean(a_snap[~np.isnan(a_snap)]))
def shrink(cs,pp,gl,cg,hp): return (c0(cs)+K*np.where(hp,c0(pp),gl))/(c0(cg)+K)
def feats(dfd,cg,hp):
    xr=shrink(dfd['c_rush'],dfd['pr_rush'],gm['pr_rush'],cg,hp); xc=shrink(dfd['c_rec'],dfd['pr_rec'],gm['pr_rec'],cg,hp)
    cpg=shrink(dfd['c_car'],dfd['pr_car'],gm['pr_car'],cg,hp); tpg=shrink(dfd['c_tgt'],dfd['pr_tgt'],gm['pr_tgt'],cg,hp)
    glp=shrink(dfd['c_gl'],dfd['pr_gl'],gm['pr_gl'],cg,hp); rzr=shrink(dfd['c_rzr'],dfd['pr_rzr'],gm['pr_rzr'],cg,hp)
    ezt=shrink(dfd['c_ezt'],dfd['pr_ezt'],gm['pr_ezt'],cg,hp); rzt=shrink(dfd['c_rzt'],dfd['pr_rzt'],gm['pr_rzt'],cg,hp)
    nv=shrink(dfd['c_scr'],dfd['pr_scr'],gm['pr_scr'],cg,hp); xtd=xr+xc; vol=cpg+tpg
    asn=np.asarray(dfd['a_snap'],float); psn=np.asarray(dfd['pr_snap'],float)
    snap=np.where(~np.isnan(asn),asn,np.where(~np.isnan(psn),psn,gsnap))
    pp=np.asarray(dfd['ppos']).astype(str)
    return xr,xc,xtd,vol,cpg,tpg,glp,rzr,ezt,rzt,snap,nv,pp
xr,xc,xtd,vol,cpg,tpg,glp,rzr,ezt,rzt,snap,nv,_=feats(df,df['c_g'],has_pr)
# team env for training rows
tg=con.sql("select season,week,posteam,t from tg").fetchnumpy()
ts=np.asarray(tg['season']);tw=np.asarray(tg['week']);tt=np.asarray(tg['posteam']);tv=np.asarray(tg['t'],float)
tkey=np.char.add(np.char.add(ts.astype(str),'_'),tt.astype(str)); trail={}
for k in np.unique(tkey):
    m=tkey==k;o=np.argsort(tw[m]);ww=tw[m][o];vv=tv[m][o];cs=np.cumsum(vv)
    for i in range(len(ww)): trail[(k,int(ww[i]))]=((cs[i]-vv[i])+K*2.4)/(i+K)
# training rows lack team key here (didn't select posteam); rebuild team_exp via a fresh pull
tr_team=con.sql("select w.posteam from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season").fetchnumpy()['posteam']
tr_team=np.asarray(tr_team)
team_exp=np.array([trail.get((f"{int(season[i])}_{tr_team[i]}",int(week[i])),2.4) for i in range(len(season))])
X=np.column_stack([xr,xc,xtd,vol,cpg,tpg,glp,rzr,ezt,rzt,snap,team_exp,nv,c_g,
                   (pos=='RB'),(pos=='WR'),(pos=='TE'),(pos=='QB')]).astype(float)
skill=np.isin(pos,['RB','WR','TE','QB']); involved=(carries+targets)>=1; hist=(c_g>=1)|has_pr
train=skill&involved&hist&(((season<SEASON))|((season==SEASON)&(week<UPCOMING)))
clf=CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,
    l2_regularization=1.0,min_samples_leaf=60,random_state=0),method='sigmoid',cv=3)
clf.fit(X[train],scored[train])
print("trained on",int(train.sum()),"completed player-games")

# ---- SNAPSHOT: current state entering the upcoming week ----
snap_df=con.sql(f"""
  with cur as (
    select pid, sum(gx_rush) c_rush,sum(gx_rec) c_rec,sum(carries) c_car,sum(targets) c_tgt,
      sum(n_gl) c_gl,sum(n_rzr) c_rzr,sum(n_ezt) c_ezt,sum(n_rzt) c_rzt,sum(scored) c_scr,
      count(*) c_g, avg(off_pct) a_snap
    from pgs where season={SEASON} and week<{UPCOMING} group by pid),
  lastteam as (
    select pid, arg_max(posteam, season*100+week) posteam
    from pg where season in ({SEASON},{SEASON-1}) group by pid)
  select coalesce(cur.pid, pr.pid) pid, po.ppos,
    coalesce(c_rush,0) c_rush,coalesce(c_rec,0) c_rec,coalesce(c_car,0) c_car,coalesce(c_tgt,0) c_tgt,
    coalesce(c_gl,0) c_gl,coalesce(c_rzr,0) c_rzr,coalesce(c_ezt,0) c_ezt,coalesce(c_rzt,0) c_rzt,
    coalesce(c_scr,0) c_scr, coalesce(c_g,0) c_g, a_snap,
    pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g,
    lt.posteam
  from cur full outer join (select * from prior where season={SEASON}) pr on cur.pid=pr.pid
  left join lastteam lt on coalesce(cur.pid,pr.pid)=lt.pid
  left join pos po on coalesce(cur.pid,pr.pid)=po.pid
""").fetchnumpy()
# player display names
nm=con.sql(f"select pid, arg_max(pname, season*100+week) nm from pg where season in ({SEASON},{SEASON-1}) group by pid").fetchnumpy()
name_map=dict(zip(np.asarray(nm['pid']).astype(str), np.asarray(nm['nm']).astype(str)))

s_has_pr=c0(snap_df['pr_g'])>0; s_cg=c0(snap_df['c_g'])
sxr,sxc,sxtd,svol,scpg,stpg,sglp,srzr,sezt,srzt,ssnap,snv,spos=feats(snap_df,snap_df['c_g'],s_has_pr)
spid=np.asarray(snap_df['pid']).astype(str); steam=np.asarray(snap_df['posteam']).astype(str)
# schedule: upcoming matchups
sch=con.sql(f"""select away_team a, home_team h from 'data/games.csv'
  where season={SEASON} and week={UPCOMING} and result is null""").fetchall()
opp_map={}; play_set=set()
for a,h in sch:
    opp_map[a]=h; opp_map[h]=a; play_set.add(a); play_set.add(h)
steam_exp=np.array([trail.get((f"{SEASON}_{steam[i]}",UPCOMING-1) if (f"{SEASON}_{steam[i]}",UPCOMING-1) in trail else (f"{SEASON}_{steam[i]}",1),2.4)
                    if steam[i] in play_set else 2.4 for i in range(len(spid))])
# simpler team_exp: latest available trailing for that team
def team_exp_now(tm):
    best=None
    for (kk,wk),v in trail.items():
        if kk==f"{SEASON}_{tm}":
            if best is None or wk>best[0]: best=(wk,v)
    return best[1] if best else 2.4
steam_exp=np.array([team_exp_now(steam[i]) for i in range(len(spid))])
Xs=np.column_stack([sxr,sxc,sxtd,svol,scpg,stpg,sglp,srzr,sezt,srzt,ssnap,steam_exp,snv,s_cg,
                    (spos=='RB'),(spos=='WR'),(spos=='TE'),(spos=='QB')]).astype(float)
ps=clf.predict_proba(Xs)[:,1]
skill_s=np.isin(spos,['RB','WR','TE','QB']); vol_ok=(scpg+stpg)>=3.0
sel=[i for i in range(len(spid)) if skill_s[i] and vol_ok[i] and steam[i] in play_set and (s_cg[i]>=1 or s_has_pr[i])]
players=[]
for i in sel:
    players.append({"name":name_map.get(spid[i],spid[i]),"team":steam[i],"opp":opp_map.get(steam[i],"—"),
        "pos":spos[i],"cpg":round(float(scpg[i]),1),"tpg":round(float(stpg[i]),1),
        "xrush":round(float(sxr[i]),3),"xrec":round(float(sxc[i]),3),"xtd":round(float(sxtd[i]),3),
        "snap":round(float(ssnap[i])*100,0),"teamExp":round(float(steam_exp[i]),2),"chance":round(float(ps[i]),4)})
players.sort(key=lambda x:-x['chance']); players=players[:80]
bt=json.load(open('out/backtest_gbm.json'))
asof=con.sql(f"select max(gameday) from 'data/games.csv' where season={SEASON} and week={UPCOMING} and result is null").fetchone()[0]
meta={"season":SEASON,"week":UPCOMING,"upcoming":True,"games":len(sch),"kickoff":str(asof),
      "rush_rates":{str(k):round(v,3) for k,v in sorted(rush.items())},
      "rec_rates":{f'{b}_{e}':round(rec.get((b,e),0),3) for b in range(5) for e in (0,1)},"bt":bt}
json.dump({"meta":meta,"players":players},open('out/board_data_upcoming.json','w'),indent=1)
print("projected",len(players),"players for",SEASON,"wk",UPCOMING)
print("top8:",[(p['name'],p['team'],p['opp'],p['chance']) for p in players[:8]])
