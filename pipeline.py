"""
v3 pipeline (shared by backtest + projection).
Adds: red-zone & goal-line CARRY-SHARE features, red-zone target share,
Vegas implied team totals (from schedule spread/total), 4-season backtest,
and injury handling (exclude Out/Doubtful + reallocate their red-zone volume).
"""
import duckdb, numpy as np, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

YEARS=[2021,2022,2023,2024,2025,2026]
def rp(s): return "read_parquet(["+",".join(f"'data/pbp_{x}.parquet'" for x in s)+"])"
def rost(s): return "read_parquet(["+",".join(f"'data/roster_{x}.parquet'" for x in s)+"])"
def snp(s):  return "read_parquet(["+",".join(f"'data/snaps_{x}.parquet'" for x in s)+"])"
K=3.0
FEATS=['xr','xc','xtd','vol','cpg','tpg','glpg','rzrpg','eztpg','rztpg',
       'rz_csh','gl_csh','rz_tsh','implied','team_exp','snap','naive','cg',
       'is_RB','is_WR','is_TE','is_QB']

def fit_rates(con,fs):
    F=rp(fs)
    rush={r[0]:r[1] for r in con.sql(f"""select case when yardline_100<=2 then 0 when yardline_100<=5 then 1
      when yardline_100<=10 then 2 when yardline_100<=20 then 3 when yardline_100<=50 then 4 else 5 end b,
      avg(rush_touchdown) rate from {F} where rush_attempt=1 and rusher_player_id is not null and yardline_100 is not null group by 1""").fetchall()}
    rec={(int(r[0]),int(r[1])):r[2] for r in con.sql(f"""select case when yardline_100<=5 then 0 when yardline_100<=10 then 1
      when yardline_100<=20 then 2 when yardline_100<=40 then 3 else 4 end b,
      case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end ez,
      avg(pass_touchdown) rate from {F} where pass_attempt=1 and receiver_player_id is not null and yardline_100 is not null group by 1,2""").fetchall()}
    return rush,rec

def build_tables(con, years, rate_years):
    rush,rec=fit_rates(con,rate_years)
    rushc=("case when yardline_100<=2 then {0} when yardline_100<=5 then {1} when yardline_100<=10 then {2} "
      "when yardline_100<=20 then {3} when yardline_100<=50 then {4} else {5} end").format(*[rush[i] for i in range(6)])
    yb="case when yardline_100<=5 then 0 when yardline_100<=10 then 1 when yardline_100<=20 then 2 when yardline_100<=40 then 3 else 4 end"
    ez="(case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end)"
    recc="case "+" ".join(f"when {yb}={b} and {ez}={e} then {rec.get((b,e),0.0)}" for b in range(5) for e in (0,1))+" else 0 end"
    ALL=rp(years)
    con.execute(f"""create or replace table plays as
      select season,week,game_id,posteam,defteam,rusher_player_id pid,rusher_player_name pname,
        1 cy,0 tg,({rushc}) rrate,0.0 crate,(yardline_100<=5)::int n_gl,(yardline_100 between 6 and 20)::int n_rzr,
        0 n_ezt,0 n_rzt,(yardline_100 between 6 and 10)::int n_i10,(yardline_100<=20)::int inrz,rush_touchdown td
        from {ALL} where rush_attempt=1 and rusher_player_id is not null and yardline_100 is not null
      union all
      select season,week,game_id,posteam,defteam,receiver_player_id,receiver_player_name,0,1,0.0,({recc}),0,0,
        ({ez}),(case when yardline_100<=20 and not({ez}=1) then 1 else 0 end),0,(yardline_100<=20)::int,pass_touchdown
        from {ALL} where pass_attempt=1 and receiver_player_id is not null and yardline_100 is not null""")
    con.execute("""create or replace table pg as
      select season,week,game_id,posteam,pid,any_value(pname) pname,sum(cy) carries,sum(tg) targets,
        sum(rrate) gx_rush,sum(crate) gx_rec,sum(n_gl) n_gl,sum(n_rzr) n_rzr,sum(n_ezt) n_ezt,sum(n_rzt) n_rzt,sum(n_i10) n_i10,
        (max(td)>0)::int scored from plays group by season,week,game_id,posteam,pid""")
    con.execute(f"""create or replace table pos as select gsis_id pid, any_value("position") ppos from {rost(years)} group by gsis_id""")
    con.execute(f"""create or replace table snap as select pl.gsis_id pid, s.game_id, max(s.offense_pct) off_pct
      from {snp(years)} s join 'data/players.parquet' pl on s.pfr_player_id=pl.pfr_id group by pl.gsis_id, s.game_id""")
    con.execute("""create or replace table pgs as select p.*, sn.off_pct from pg p left join snap sn on p.pid=sn.pid and p.game_id=sn.game_id""")
    # team red-zone carry/target totals per game, then trailing
    con.execute("""create or replace table teamrz as
      select season,week,posteam, sum(n_gl) tgl, sum(n_gl+n_rzr) trz, sum(n_ezt+n_rzt) trt,
        sum(inrz) rz_pl, sum(case when inrz=1 and td=1 then 1 else 0 end) rz_td
      from plays group by season,week,posteam""")
    con.execute("""create or replace table teamrz_trail as
      select season,week,posteam,
        coalesce(sum(tgl) over w,0) tgl_tr, coalesce(sum(trz) over w,0) trz_tr, coalesce(sum(trt) over w,0) trt_tr, coalesce(sum(rz_pl) over w,0) rzpl_tr,
        coalesce(sum(rz_td) over w,0) rztd_tr, count(*) over w tg_g
      from teamrz window w as (partition by posteam,season order by week rows between unbounded preceding and 1 preceding)""")
    # Vegas implied team totals from schedule
    con.execute("""create or replace table vegas as
      with g as (select game_id, home_team, away_team, total_line, spread_line from 'data/games.csv'
                 where total_line is not null and spread_line is not null)
      select game_id, home_team team, total_line/2.0 + spread_line/2.0 implied from g
      union all select game_id, away_team, total_line/2.0 - spread_line/2.0 from g""")
    con.execute("""create or replace table pgw as
      select p.*, po.ppos,
        coalesce(sum(gx_rush) over w,0) c_rush,coalesce(sum(gx_rec) over w,0) c_rec,
        coalesce(sum(carries) over w,0) c_car,coalesce(sum(targets) over w,0) c_tgt,
        coalesce(sum(n_gl) over w,0) c_gl,coalesce(sum(n_rzr) over w,0) c_rzr,
        coalesce(sum(n_ezt) over w,0) c_ezt,coalesce(sum(n_rzt) over w,0) c_rzt, coalesce(sum(n_i10) over w,0) c_i10,
        coalesce(sum(scored) over w,0) c_scr,count(*) over w c_g, avg(off_pct) over w a_snap
      from pgs p left join pos po using(pid)
      window w as (partition by pid,season order by week rows between unbounded preceding and 1 preceding)""")
    con.execute("""create or replace table prior as
      select pid,season+1 season,avg(gx_rush) pr_rush,avg(gx_rec) pr_rec,avg(carries) pr_car,avg(targets) pr_tgt,
        avg(n_gl) pr_gl,avg(n_rzr) pr_rzr,avg(n_ezt) pr_ezt,avg(n_rzt) pr_rzt,avg(n_i10) pr_i10,avg(scored) pr_scr,avg(off_pct) pr_snap,
        count(*) pr_g from pgs group by pid,season""")
    con.execute("""create or replace table tg as select season,week,posteam,sum(sa) t from (
        select season,week,game_id,posteam,(max(td)>0)::int sa from plays group by season,week,game_id,posteam,pid)
        group by season,week,posteam""")

def team_env(con):
    tg=con.sql("select season,week,posteam,t from tg").fetchnumpy()
    ts=np.asarray(tg['season']);tw=np.asarray(tg['week']);tt=np.asarray(tg['posteam']);tv=np.asarray(tg['t'],float)
    tkey=np.char.add(np.char.add(ts.astype(str),'_'),tt.astype(str)); trail={}
    for k in np.unique(tkey):
        m=tkey==k;o=np.argsort(tw[m]);ww=tw[m][o];vv=tv[m][o];cs=np.cumsum(vv)
        for i in range(len(ww)): trail[(k,int(ww[i]))]=((cs[i]-vv[i])+K*2.4)/(i+K)
    return trail
