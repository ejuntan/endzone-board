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
       'rz_csh','gl_csh','rz_tsh','exp_gl_td','implied','team_exp','snap','naive','cg',
       'cpg3','tpg3','glp3','snap3','trend_car','trend_gl','trend_tgt',
       'n_sep','n_yacoe','n_cush','n_catchp','n_eff','n_ryoe','n_box8','n_rpoe',
       'is_RB','is_WR','is_TE','is_QB']
# Next Gen Stats trailing form features (receiving separation/YAC-over-exp/cushion/catch%,
# rushing efficiency/yards-over-exp/%8+box/rush%-over-exp). Fallback: trailing -> prior season -> global.
NGS=['n_sep','n_yacoe','n_cush','n_catchp','n_eff','n_ryoe','n_box8','n_rpoe']
def ngs_global_means(df):
    import numpy as _np
    return {n: float(_np.nanmean(_np.asarray(df[n],float))) for n in NGS}
def ngs_cols(df, gmeans):
    import numpy as _np
    out={}
    for n in NGS:
        v=_np.asarray(df[n],float); pv=_np.asarray(df['p'+n[1:]],float)
        out[n]=_np.where(~_np.isnan(v),v,_np.where(~_np.isnan(pv),pv,gmeans[n]))
    return out

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
    # deterministic position: most-frequent listed position, ties broken alphabetically
    con.execute(f"""create or replace table pos as
      select pid, ppos from (
        select gsis_id pid, "position" ppos,
          row_number() over (partition by gsis_id order by count(*) desc, "position") rn
        from {rost(years)} where "position" is not null group by gsis_id, "position") where rn=1""")
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
        coalesce(sum(scored) over w,0) c_scr,count(*) over w c_g, avg(off_pct) over w a_snap,
        coalesce(sum(carries) over w3,0) c_car_l3, coalesce(sum(targets) over w3,0) c_tgt_l3,
        coalesce(sum(n_gl) over w3,0) c_gl_l3, coalesce(sum(n_rzr) over w3,0) c_rzr_l3,
        count(*) over w3 c_g_l3, avg(off_pct) over w3 a_snap_l3
      from pgs p left join pos po using(pid)
      window w as (partition by pid,season order by week rows between unbounded preceding and 1 preceding),
             w3 as (partition by pid,season order by week rows between 3 preceding and 1 preceding)""")
    con.execute("""create or replace table prior as
      select pid,season+1 season,avg(gx_rush) pr_rush,avg(gx_rec) pr_rec,avg(carries) pr_car,avg(targets) pr_tgt,
        avg(n_gl) pr_gl,avg(n_rzr) pr_rzr,avg(n_ezt) pr_ezt,avg(n_rzt) pr_rzt,avg(n_i10) pr_i10,avg(scored) pr_scr,avg(off_pct) pr_snap,
        count(*) pr_g from pgs group by pid,season""")
    # Next Gen Stats: merge receiving+rushing weekly to one row per player-week, then trail + prior-season
    con.execute("""create or replace table ngm as
      select pid,season,week, max(sep) sep,max(yacoe) yacoe,max(cush) cush,max(catchp) catchp,
        max(eff) eff,max(ryoe) ryoe,max(box8) box8,max(rpoe) rpoe from (
        select player_gsis_id pid,season,week,avg_separation sep,avg_yac_above_expectation yacoe,
               avg_cushion cush,catch_percentage catchp,NULL::double eff,NULL::double ryoe,NULL::double box8,NULL::double rpoe
        from read_parquet('data/ngs_receiving.parquet') where week>0
        union all
        select player_gsis_id,season,week,NULL,NULL,NULL,NULL,
               efficiency,rush_yards_over_expected_per_att,percent_attempts_gte_eight_defenders,rush_pct_over_expected
        from read_parquet('data/ngs_rushing.parquet') where week>0)
      group by pid,season,week""")
    con.execute("""create or replace table ngs_trail as
      select pid,season,week,
        avg(sep) over w n_sep, avg(yacoe) over w n_yacoe, avg(cush) over w n_cush, avg(catchp) over w n_catchp,
        avg(eff) over w n_eff, avg(ryoe) over w n_ryoe, avg(box8) over w n_box8, avg(rpoe) over w n_rpoe
      from ngm window w as (partition by pid,season order by week rows between unbounded preceding and 1 preceding)""")
    con.execute("""create or replace table ngs_prior as
      select pid, season+1 season, avg(sep) p_sep,avg(yacoe) p_yacoe,avg(cush) p_cush,avg(catchp) p_catchp,
        avg(eff) p_eff,avg(ryoe) p_ryoe,avg(box8) p_box8,avg(rpoe) p_rpoe
      from ngm group by pid,season""")
    con.execute("""create or replace table tg as select season,week,posteam,sum(sa) t from (
        select season,week,game_id,posteam,(max(td)>0)::int sa from plays group by season,week,game_id,posteam,pid)
        group by season,week,posteam""")

# ---- availability (play-probability) model: stage 1 of the two-stage board ----
# stage-1 features (predict >=1 offensive touch given pregame-eligible)
STAGE1=['stat','prac','snap','snap3','pg3','cg','vol','hp','is_RB','is_WR','is_TE','is_QB']
def status_code(s):
    return {'':0,'None':0,'Questionable':1,'Doubtful':2}.get(str(s),0)
def practice_code(s):
    s=str(s)
    return 2 if 'Did Not' in s else 1 if 'Limited' in s else 0

def build_grid(con, years, inj_files):
    """Dense pregame player-week grid over `years` with exact trailing usage,
    last-3 recency, whether the player actually played, and injury/practice status.
    `inj_files` = list of injuries parquet paths covering `years`."""
    import numpy as _np
    INJ="read_parquet(["+",".join(f"'{f}'" for f in inj_files)+"])"
    con.execute(f"""create or replace table grid as
    with span as (select pid,season,min(week) mn,max(week) mx, arg_max(posteam,week) tm from pg group by pid,season),
    wks as (select unnest(generate_series(1,22)) wknum),
    pw as (select s.pid,s.season,x.wknum,s.tm from span s join wks x on x.wknum between s.mn and s.mx),
    sched as (select distinct season,week,posteam,game_id from pg),
    elig as (select pw.pid,pw.season,pw.wknum,pw.tm posteam, sc.game_id
             from pw join sched sc on pw.season=sc.season and pw.wknum=sc.week and pw.tm=sc.posteam),
    joined as (
      select e.pid,e.season,e.wknum,e.posteam,e.game_id,
        coalesce(g.carries,0) carries,coalesce(g.targets,0) targets,
        coalesce(g.scored,0) scored, gs.off_pct, (g.pid is not null)::int played
      from elig e
      left join pg g on e.pid=g.pid and e.season=g.season and e.wknum=g.week
      left join pgs gs on e.pid=gs.pid and e.season=gs.season and e.wknum=gs.week)
    select *,
      coalesce(sum(carries) over w,0) c_car,coalesce(sum(targets) over w,0) c_tgt,
      coalesce(sum(played) over w,0) c_g, avg(off_pct) over w a_snap,
      avg(off_pct) over w3 a_snap_l3, coalesce(sum(played) over w3,0) p_g_l3
    from joined
    window w  as (partition by pid,season order by wknum rows between unbounded preceding and 1 preceding),
           w3 as (partition by pid,season order by wknum rows between 3 preceding and 1 preceding)""")
    return con.sql(f"""select gr.season,gr.wknum,gr.posteam,po.ppos,gr.played,gr.scored,
        gr.c_car,gr.c_tgt,gr.c_g,gr.a_snap,gr.a_snap_l3,gr.p_g_l3,
        pr.pr_car,pr.pr_tgt,pr.pr_g, inj.status, inj.practice
      from grid gr left join pos po using(pid)
      left join prior pr on gr.pid=pr.pid and gr.season=pr.season
      left join (select season,week,gsis_id pid,any_value(report_status) status,any_value(practice_status) practice
                 from {INJ} group by 1,2,3) inj on gr.pid=inj.pid and gr.season=inj.season and gr.wknum=inj.week
      order by gr.season,gr.wknum,gr.pid
    """).fetchnumpy()

def stage1_matrix(d, gsnap):
    """Build the stage-1 feature matrix from a grid/snapshot numpy frame."""
    import numpy as _np
    def c0(a): a=_np.asarray(a,float); a[_np.isnan(a)]=0.0; return a
    hp=c0(d['pr_g'])>0; cg=c0(d['c_g']); pos=d['ppos'].astype(str)
    aa=_np.asarray(d['a_snap'],float); snap=_np.where(~_np.isnan(aa),aa,gsnap)
    a3=_np.asarray(d['a_snap_l3'],float); snap3=_np.where(~_np.isnan(a3),a3,snap)
    pg3=c0(d['p_g_l3'])
    def sh(cs,pp): return (c0(cs)+K*_np.where(hp,c0(pp),0.0))/(cg+K)
    vol=sh(d['c_car'],d['pr_car'])+sh(d['c_tgt'],d['pr_tgt'])
    stat=_np.array([status_code(s) for s in d['status']],float)
    prac=_np.array([practice_code(s) for s in d['practice']],float)
    cols=[stat,prac,snap,snap3,pg3,cg,vol,hp.astype(float),
          (pos=='RB').astype(float),(pos=='WR').astype(float),(pos=='TE').astype(float),(pos=='QB').astype(float)]
    return _np.column_stack(cols).astype(float)

def team_env(con):
    tg=con.sql("select season,week,posteam,t from tg").fetchnumpy()
    ts=np.asarray(tg['season']);tw=np.asarray(tg['week']);tt=np.asarray(tg['posteam']);tv=np.asarray(tg['t'],float)
    tkey=np.char.add(np.char.add(ts.astype(str),'_'),tt.astype(str)); trail={}
    for k in np.unique(tkey):
        m=tkey==k;o=np.argsort(tw[m]);ww=tw[m][o];vv=tv[m][o];cs=np.cumsum(vv)
        for i in range(len(ww)): trail[(k,int(ww[i]))]=((cs[i]-vv[i])+K*2.4)/(i+K)
    return trail
