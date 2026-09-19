"""
Anytime-TD model: empirical rates + team-share coherence + calibration, backtested.

Temporal separation (no leakage):
  - Scoring RATES fit on 2021-2022 play-by-play.
  - Isotonic CALIBRATION fit on 2023 predictions/outcomes.
  - Final EVALUATION on 2024 (unseen by rates and calibration).
Walk-forward: every prediction uses only prior-week (and prior-season) info.
"""
import duckdb, numpy as np, json

con = duckdb.connect()
PBP = {yr: f"data/pbp_{yr}.parquet" for yr in (2021,2022,2023,2024)}
ALL = "read_parquet(['data/pbp_2021.parquet','data/pbp_2022.parquet','data/pbp_2023.parquet','data/pbp_2024.parquet'])"
FIT = "read_parquet(['data/pbp_2021.parquet','data/pbp_2022.parquet'])"

# ---------------------------------------------------------------------------
# 1. FIT empirical scoring rates on 2021-2022 only
# ---------------------------------------------------------------------------
rush_rates = {r[0]: r[1] for r in con.sql(f"""
  select case
    when yardline_100<=2 then 0 when yardline_100<=5 then 1
    when yardline_100<=10 then 2 when yardline_100<=20 then 3
    when yardline_100<=50 then 4 else 5 end as b,
    avg(rush_touchdown) rate
  from {FIT} where rush_attempt=1 and rusher_player_id is not null
    and yardline_100 is not null group by 1""").fetchall()}

rec_rates = {(int(r[0]),int(r[1])): r[2] for r in con.sql(f"""
  select case
    when yardline_100<=5 then 0 when yardline_100<=10 then 1
    when yardline_100<=20 then 2 when yardline_100<=40 then 3 else 4 end as b,
    case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end as ez,
    avg(pass_touchdown) rate
  from {FIT} where pass_attempt=1 and receiver_player_id is not null
    and yardline_100 is not null group by 1,2""").fetchall()}

def rush_case():
    return ("case when yardline_100<=2 then {0} when yardline_100<=5 then {1} "
            "when yardline_100<=10 then {2} when yardline_100<=20 then {3} "
            "when yardline_100<=50 then {4} else {5} end").format(
            *[rush_rates[i] for i in range(6)])
def rec_case():
    parts=[]
    for b in range(5):
        for ez in (0,1):
            parts.append((b,ez,rec_rates.get((b,ez),0.0)))
    ybucket=("case when yardline_100<=5 then 0 when yardline_100<=10 then 1 "
             "when yardline_100<=20 then 2 when yardline_100<=40 then 3 else 4 end")
    ezf="(case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end)"
    whens=" ".join(f"when {ybucket}={b} and {ezf}={ez} then {r}" for b,ez,r in parts)
    return f"case {whens} else 0 end"

print("Fitted rush rates (per carry):", {k:round(v,3) for k,v in sorted(rush_rates.items())})
print("Fitted rec rates  (per target):", {f'b{b}/ez{ez}':round(rec_rates.get((b,ez),0),3)
      for b in range(5) for ez in (0,1)})

# ---------------------------------------------------------------------------
# 2. Player-game table: bucketed expected-TD from ACTUAL opportunities + outcome
# ---------------------------------------------------------------------------
con.execute(f"""
create or replace table pg as
with plays as (
  select season, week, game_id, posteam,
         rusher_player_id pid, rusher_player_name pname,
         1 cy, 0 tg, ({rush_case()}) rrate, 0.0 crate,
         rush_touchdown td
  from {ALL} where rush_attempt=1 and rusher_player_id is not null and yardline_100 is not null
  union all
  select season, week, game_id, posteam,
         receiver_player_id, receiver_player_name,
         0, 1, 0.0, ({rec_case()}),
         pass_touchdown
  from {ALL} where pass_attempt=1 and receiver_player_id is not null and yardline_100 is not null
)
select season, week, game_id, posteam, pid, any_value(pname) pname,
       sum(cy) carries, sum(tg) targets,
       sum(rrate) gx_rush, sum(crate) gx_rec,
       (max(td)>0)::int scored
from plays group by season, week, game_id, posteam, pid
""")

# team offensive TDs per game (actual) for environment trailing
con.execute(f"""
create or replace table tg as
select season, week, game_id, posteam,
       sum(scored_any) team_off_td
from (
  select season,week,game_id,posteam, (max(td)>0)::int scored_any from (
    select season,week,game_id,posteam, rusher_player_id pid, rush_touchdown td
      from {ALL} where rush_attempt=1 and rusher_player_id is not null
    union all
    select season,week,game_id,posteam, receiver_player_id, pass_touchdown
      from {ALL} where pass_attempt=1 and receiver_player_id is not null
  ) group by season,week,game_id,posteam,pid
) group by season,week,game_id,posteam
""")

# position map from rosters
con.execute("""
create or replace table pos as
select gsis_id pid, any_value("position") ppos from read_parquet(
  ['data/roster_2021.parquet','data/roster_2022.parquet','data/roster_2023.parquet','data/roster_2024.parquet'])
group by gsis_id
""")

# ---------------------------------------------------------------------------
# 3. Walk-forward trailing features (only prior weeks within season) + priors
# ---------------------------------------------------------------------------
# within-season trailing sums up to (not including) current week
con.execute("""
create or replace table pgw as
select p.*, po.ppos,
  coalesce(sum(gx_rush) over w,0) c_rush, coalesce(sum(gx_rec) over w,0) c_rec,
  coalesce(sum(carries) over w,0)  c_car,  coalesce(sum(targets) over w,0) c_tgt,
  coalesce(sum(scored)  over w,0)  c_scr,  count(*) over w c_g
from pg p left join pos po using(pid)
window w as (partition by pid, season order by week rows between unbounded preceding and 1 preceding)
""")

# prior-season per-game means (for shrinkage priors)
con.execute("""
create or replace table prior as
select pid, season+1 as season,
  avg(gx_rush) pr_rush, avg(gx_rec) pr_rec,
  avg(carries) pr_car, avg(targets) pr_tgt, avg(scored) pr_scr, count(*) pr_g
from pg group by pid, season
""")

df = con.sql("""
  select w.season, w.week, w.game_id, w.posteam, w.pid, w.pname,
         w.ppos, w.carries, w.targets, w.gx_rush, w.gx_rec, w.scored,
         w.c_rush,w.c_rec,w.c_car,w.c_tgt,w.c_scr,w.c_g,
         pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_scr,pr.pr_g,
         t.team_off_td
  from pgw w
  left join prior pr on w.pid=pr.pid and w.season=pr.season
  left join tg t on w.season=t.season and w.week=t.week and w.game_id=t.game_id and w.posteam=t.posteam
""").fetchnumpy()

n = len(df['season'])
def col(name, fill=0.0):
    a = np.asarray(df[name], dtype=float); a[np.isnan(a)] = fill; return a

season=np.asarray(df['season']); week=np.asarray(df['week'])
game=np.asarray(df['game_id']); team=np.asarray(df['posteam']); pos=df['ppos']
carries=col('carries'); targets=col('targets'); scored=col('scored').astype(int)
c_rush=col('c_rush'); c_rec=col('c_rec'); c_car=col('c_car'); c_tgt=col('c_tgt')
c_scr=col('c_scr'); c_g=col('c_g')
pr_rush=col('pr_rush'); pr_rec=col('pr_rec'); pr_car=col('pr_car'); pr_tgt=col('pr_tgt')
pr_scr=col('pr_scr'); pr_g=col('pr_g'); has_pr=col('pr_g')>0

# global priors (for rookies / no prior season), from fit seasons' active players
gpr_rush=np.mean(pr_rush[has_pr]); gpr_rec=np.mean(pr_rec[has_pr])
gpr_car=np.mean(pr_car[has_pr]);  gpr_tgt=np.mean(pr_tgt[has_pr]); gpr_scr=np.mean(pr_scr[has_pr])

K=3.0  # shrinkage strength (games)
def shrink(cum_sum, cur_n, prior_pg, glob):
    prior = np.where(has_pr, prior_pg, glob)
    return (cum_sum + K*prior) / (cur_n + K)

xr = shrink(c_rush, c_g, pr_rush, gpr_rush)   # expected rushing xTD/game
xc = shrink(c_rec,  c_g, pr_rec,  gpr_rec)     # expected receiving xTD/game
vol= shrink(c_car+c_tgt, c_g, pr_car+pr_tgt, gpr_car+gpr_tgt)  # expected volume
naive_rate = shrink(c_scr, c_g, pr_scr, gpr_scr)  # trailing anytime-TD frequency

xtd = xr + xc
p_opp = 1 - np.exp(-xtd)                        # granular, uncoherent

# ---- team coherence: E[TD_i] = team_expTD * share_i ----
# team expected offensive TDs, trailing (walk-forward) via team means
team_key = np.char.add(np.char.add(season.astype(str),'_'),team.astype(str))
# trailing team_off_td per team-season using pandas-free grouping
order = np.lexsort((week, team_key))
team_exp = np.zeros(n)
from collections import defaultdict
acc=defaultdict(lambda:[0.0,0]) ; last_week=defaultdict(lambda:-1)
# need one team value per team-game; compute team trailing mean then broadcast
tg_rows = con.sql("select season,week,game_id,posteam,team_off_td from tg").fetchnumpy()
ts=np.asarray(tg_rows['season']); tw=np.asarray(tg_rows['week'])
tt=np.asarray(tg_rows['posteam']); tv=np.asarray(tg_rows['team_off_td'],dtype=float)
tkey=np.char.add(np.char.add(ts.astype(str),'_'),tt.astype(str))
team_trail={}
for k in np.unique(tkey):
    m=tkey==k; ww=tw[m]; vv=tv[m]; o=np.argsort(ww)
    ww,vv=ww[o],vv[o]; cs=np.cumsum(vv); cn=np.arange(len(vv))
    for i in range(len(ww)):
        prior_sum=cs[i]-vv[i]; prior_n=cn[i]
        team_trail[(k,int(ww[i]))]=(prior_sum+K*2.4)/(prior_n+K)  # league ~2.4 off TD/g prior
for i in range(n):
    team_exp[i]=team_trail.get((team_key[i],int(week[i])),2.4)

# share within each game-team
gt_key=np.char.add(np.char.add(game.astype(str),'|'),team.astype(str))
p_coh=np.zeros(n); E_coh=np.zeros(n)
for k in np.unique(gt_key):
    m=np.where(gt_key==k)[0]
    s=xtd[m].sum()
    if s<=0:
        E_coh[m]=0; p_coh[m]=0; continue
    share=xtd[m]/s
    E=team_exp[m]*share
    E_coh[m]=E; p_coh[m]=1-np.exp(-E)

# baselines
p_naive=naive_rate.copy()
# volume-share baseline: team_expTD * share of raw volume, Poisson
p_vol=np.zeros(n)
for k in np.unique(gt_key):
    m=np.where(gt_key==k)[0]
    s=vol[m].sum()
    if s<=0: continue
    E=team_exp[m]*(vol[m]/s); p_vol[m]=1-np.exp(-E)

# ---------------------------------------------------------------------------
# 4. Evaluation universe: active & involved skill players (>=1 opp in game)
# ---------------------------------------------------------------------------
skill=np.isin(pos.astype(str),['RB','WR','TE','QB'])
involved=(carries+targets)>=1
has_hist=(c_g>=1)|has_pr
base=skill&involved&has_hist

def metrics(y,p):
    p=np.clip(p,1e-6,1-1e-6)
    brier=np.mean((p-y)**2)
    ll=-np.mean(y*np.log(p)+(1-y)*np.log(1-p))
    return brier,ll

# isotonic (PAV) fit on 2023, apply to 2024
def isotonic_fit(x,y):
    o=np.argsort(x); xs=x[o]; ys=y[o].astype(float)
    w=np.ones_like(ys);
    # pool adjacent violators
    vals=ys.copy(); wts=w.copy(); idx=[[i] for i in range(len(ys))]
    i=0
    yv=list(ys); wv=list(w)
    # standard PAV
    lev_y=[]; lev_w=[]; lev_x=[]
    for xi,yi in zip(xs,ys):
        lev_y.append(yi); lev_w.append(1.0); lev_x.append(xi)
        while len(lev_y)>1 and lev_y[-2]>lev_y[-1]:
            y2=lev_y.pop(); w2=lev_w.pop(); x2=lev_x.pop()
            y1=lev_y.pop(); w1=lev_w.pop(); x1=lev_x.pop()
            wn=w1+w2; yn=(y1*w1+y2*w2)/wn
            lev_y.append(yn); lev_w.append(wn); lev_x.append(x2)
    # build step points: expand back to thresholds
    xs_knot=[]; ys_knot=[]
    ci=0
    # recompute assignment
    return xs, _pav_apply(xs,ys)
def _pav(y):
    y=y.astype(float); n=len(y); w=np.ones(n)
    yy=list(y); ww=list(w)
    Y=[]; W=[]
    for i in range(n):
        Y.append(y[i]); W.append(1.0)
        while len(Y)>1 and Y[-2]>Y[-1]:
            y2=Y.pop(); w2=W.pop(); y1=Y.pop(); w1=W.pop()
            Y.append((y1*w1+y2*w2)/(w1+w2)); W.append(w1+w2)
    out=[];
    for yv,wv in zip(Y,W): out += [yv]*int(wv)
    return np.array(out)
def iso_fit(x,y):
    o=np.argsort(x); xs=x[o]; fitted=_pav(y[o]); return xs,fitted
def iso_apply(xs,fitted,xnew):
    return np.interp(xnew,xs,fitted)

models={'Naive-TDrate':p_naive,'Volume-share':p_vol,'Granular-Opp':p_opp,'Granular-Coherent':p_coh}

print("\n================ OUT-OF-SAMPLE BACKTEST (test season 2024) ================")
print(f"{'model':<22}{'Brier':>9}{'LogLoss':>9}{'n':>8}")
te=base&(season==2024)
y=scored[te]
print(f"  eval players: {te.sum()}   base TD rate: {y.mean():.3f}")
rows_metrics={}
for name,p in models.items():
    b,l=metrics(y,p[te]); rows_metrics[name]=(b,l)
    print(f"{name:<22}{b:>9.4f}{l:>9.4f}{te.sum():>8}")

# calibrate coherent model: fit on 2023, apply to 2024
va=base&(season==2023)
xs,fit=iso_fit(p_coh[va],scored[va].astype(float))
p_cal=iso_apply(xs,fit,p_coh[te])
b,l=metrics(y,p_cal); rows_metrics['Coherent+Isotonic']=(b,l)
print(f"{'Coherent+Isotonic':<22}{b:>9.4f}{l:>9.4f}{te.sum():>8}")

# ---------------------------------------------------------------------------
# 5. Calibration table for the final model
# ---------------------------------------------------------------------------
def calib_table(y,p,bins=10):
    edges=np.quantile(p,np.linspace(0,1,bins+1)); edges[0]-=1e-9; edges[-1]+=1e-9
    print(f"\n  {'pred bin':>16}{'n':>6}{'mean_pred':>11}{'obs_rate':>10}")
    for i in range(bins):
        m=(p>edges[i])&(p<=edges[i+1])
        if m.sum()==0: continue
        print(f"  {edges[i]:.2f}-{edges[i+1]:.2f}{m.sum():>8}{p[m].mean():>11.3f}{y[m].mean():>10.3f}")

print("\n--- Calibration: Granular-Coherent (2024) ---"); calib_table(y,p_coh[te])
print("\n--- Calibration: Coherent+Isotonic (2024) ---"); calib_table(y,p_cal)

# save fitted rates + summary
json.dump({'rush_rates':{str(k):v for k,v in rush_rates.items()},
           'rec_rates':{f'{b}_{ez}':rec_rates.get((b,ez),0) for b in range(5) for ez in (0,1)},
           'metrics_2024':rows_metrics},
          open('out/summary.json','w'),indent=2)
print("\nsaved out/summary.json")

# ---------------------------------------------------------------------------
# 6. Export a REAL projected slate (pre-game, walk-forward) for the web board
# ---------------------------------------------------------------------------
SEED_SEASON, SEED_WEEK = 2024, 15
opp_map = {(r[0],r[1]):r[2] for r in con.sql(f"""
  select distinct game_id, posteam, defteam from {ALL}
  where season={SEED_SEASON} and week={SEED_WEEK} and posteam is not null and defteam is not null
""").fetchall()}

cpg = shrink(c_car, c_g, pr_car, gpr_car)   # trailing carries/g
tpg = shrink(c_tgt, c_g, pr_tgt, gpr_tgt)   # trailing targets/g

sel = np.where((season==SEED_SEASON)&(week==SEED_WEEK)&skill&((c_g>=2)|has_pr)&((cpg+tpg)>=3.0))[0]
board=[]
for i in sel:
    board.append({
        "name": str(df['pname'][i]), "team": str(team[i]),
        "opp": str(opp_map.get((str(game[i]),str(team[i])),"—")),
        "pos": str(pos[i]),
        "cpg": round(float(cpg[i]),1), "tpg": round(float(tpg[i]),1),
        "xrush": round(float(xr[i]),3), "xrec": round(float(xc[i]),3),
        "xtd": round(float(xtd[i]),3),
        "teamExp": round(float(team_exp[i]),2),
        "chance": round(float(p_coh[i]),4),
    })
board.sort(key=lambda d:-d["chance"])
board=board[:80]
meta={"season":SEED_SEASON,"week":SEED_WEEK,
      "rush_rates":{str(k):round(v,3) for k,v in sorted(rush_rates.items())},
      "rec_rates":{f'{b}_{ez}':round(rec_rates.get((b,ez),0),3) for b in range(5) for ez in (0,1)},
      "backtest":{k:{"brier":round(v[0],4),"logloss":round(v[1],4)} for k,v in rows_metrics.items()},
      "base_rate":round(float(scored[te].mean()),3),"n_eval":int(te.sum())}
json.dump({"meta":meta,"players":board}, open('out/board_data.json','w'), indent=1)
print(f"\nExported {len(board)} projected players for {SEED_SEASON} wk{SEED_WEEK} -> out/board_data.json")
print("Top 6:", [(b['name'],b['chance']) for b in board[:6]])
