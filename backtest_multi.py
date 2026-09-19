"""
Multi-season walk-forward backtest.
For each TEST season, scoring rates are fit ONLY on prior seasons (expanding window):
   test 2022 <- rates from 2021
   test 2023 <- rates from 2021-22
   test 2024 <- rates from 2021-23
Every within-season prediction is walk-forward (prior weeks only).
Reports: Brier, log loss, AUC, calibration (ECE), Brier skill score, tier hit-rates.
"""
import duckdb, numpy as np
con = duckdb.connect()
def rp(seasons): return "read_parquet([" + ",".join(f"'data/pbp_{s}.parquet'" for s in seasons) + "])"
def rr_roster(): return "read_parquet(['data/roster_2021.parquet','data/roster_2022.parquet','data/roster_2023.parquet','data/roster_2024.parquet'])"
K=3.0

def fit_rates(fit_seasons):
    FIT=rp(fit_seasons)
    rush={r[0]:r[1] for r in con.sql(f"""select case when yardline_100<=2 then 0 when yardline_100<=5 then 1
        when yardline_100<=10 then 2 when yardline_100<=20 then 3 when yardline_100<=50 then 4 else 5 end b,
        avg(rush_touchdown) rate from {FIT} where rush_attempt=1 and rusher_player_id is not null
        and yardline_100 is not null group by 1""").fetchall()}
    rec={(int(r[0]),int(r[1])):r[2] for r in con.sql(f"""select case when yardline_100<=5 then 0 when yardline_100<=10 then 1
        when yardline_100<=20 then 2 when yardline_100<=40 then 3 else 4 end b,
        case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end ez,
        avg(pass_touchdown) rate from {FIT} where pass_attempt=1 and receiver_player_id is not null
        and yardline_100 is not null group by 1,2""").fetchall()}
    return rush,rec

def rush_case(rush):
    return ("case when yardline_100<=2 then {0} when yardline_100<=5 then {1} when yardline_100<=10 then {2} "
            "when yardline_100<=20 then {3} when yardline_100<=50 then {4} else {5} end").format(*[rush[i] for i in range(6)])
def rec_case(rec):
    yb=("case when yardline_100<=5 then 0 when yardline_100<=10 then 1 when yardline_100<=20 then 2 "
        "when yardline_100<=40 then 3 else 4 end")
    ez="(case when air_yards is not null and air_yards>=yardline_100 then 1 else 0 end)"
    whens=" ".join(f"when {yb}={b} and {ez}={e} then {rec.get((b,e),0.0)}" for b in range(5) for e in (0,1))
    return f"case {whens} else 0 end"

def predict_season(test_season):
    fit_seasons=[s for s in range(2021,test_season)]
    rush,rec=fit_rates(fit_seasons)
    need=[s for s in range(2021,test_season+1)]      # test season + history for trailing/priors
    ALL=rp(need)
    con.execute(f"""create or replace table pg as
      with plays as (
        select season,week,game_id,posteam,rusher_player_id pid,rusher_player_name pname,
               1 cy,0 tg,({rush_case(rush)}) rrate,0.0 crate,rush_touchdown td
        from {ALL} where rush_attempt=1 and rusher_player_id is not null and yardline_100 is not null
        union all
        select season,week,game_id,posteam,receiver_player_id,receiver_player_name,
               0,1,0.0,({rec_case(rec)}),pass_touchdown
        from {ALL} where pass_attempt=1 and receiver_player_id is not null and yardline_100 is not null)
      select season,week,game_id,posteam,pid,any_value(pname) pname,sum(cy) carries,sum(tg) targets,
             sum(rrate) gx_rush,sum(crate) gx_rec,(max(td)>0)::int scored
      from plays group by season,week,game_id,posteam,pid""")
    con.execute(f"""create or replace table tg as
      select season,week,game_id,posteam,sum(sa) team_off_td from (
        select season,week,game_id,posteam,(max(td)>0)::int sa from (
          select season,week,game_id,posteam,rusher_player_id pid,rush_touchdown td from {ALL}
            where rush_attempt=1 and rusher_player_id is not null
          union all select season,week,game_id,posteam,receiver_player_id,pass_touchdown from {ALL}
            where pass_attempt=1 and receiver_player_id is not null)
        group by season,week,game_id,posteam,pid) group by season,week,game_id,posteam""")
    con.execute(f"""create or replace table pos as
      select gsis_id pid, any_value("position") ppos from {rr_roster()} group by gsis_id""")
    con.execute("""create or replace table pgw as
      select p.*, po.ppos,
        coalesce(sum(gx_rush) over w,0) c_rush, coalesce(sum(gx_rec) over w,0) c_rec,
        coalesce(sum(carries) over w,0) c_car, coalesce(sum(targets) over w,0) c_tgt,
        coalesce(sum(scored) over w,0) c_scr, count(*) over w c_g
      from pg p left join pos po using(pid)
      window w as (partition by pid,season order by week rows between unbounded preceding and 1 preceding)""")
    con.execute("""create or replace table prior as
      select pid, season+1 season, avg(gx_rush) pr_rush, avg(gx_rec) pr_rec,
        avg(carries) pr_car, avg(targets) pr_tgt, avg(scored) pr_scr, count(*) pr_g
      from pg group by pid,season""")
    df=con.sql("""select w.season,w.week,w.game_id,w.posteam,w.pid,w.pname,w.ppos,w.carries,w.targets,w.scored,
        w.c_rush,w.c_rec,w.c_car,w.c_tgt,w.c_scr,w.c_g,
        pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_scr,pr.pr_g, t.team_off_td
      from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season
      left join tg t on w.season=t.season and w.week=t.week and w.game_id=t.game_id and w.posteam=t.posteam
    """).fetchnumpy()
    def c(n): a=np.asarray(df[n],dtype=float); a[np.isnan(a)]=0.0; return a
    season=np.asarray(df['season']); week=np.asarray(df['week'])
    game=np.asarray(df['game_id']); team=np.asarray(df['posteam']); pos=df['ppos']
    carries=c('carries'); targets=c('targets'); scored=c('scored').astype(int)
    c_rush=c('c_rush');c_rec=c('c_rec');c_car=c('c_car');c_tgt=c('c_tgt');c_scr=c('c_scr');c_g=c('c_g')
    pr_rush=c('pr_rush');pr_rec=c('pr_rec');pr_car=c('pr_car');pr_tgt=c('pr_tgt');pr_scr=c('pr_scr')
    has_pr=c('pr_g')>0
    g=lambda a:np.mean(a[has_pr])
    def shrink(cs,cn,pp,gl): return (cs+K*np.where(has_pr,pp,gl))/(cn+K)
    xr=shrink(c_rush,c_g,pr_rush,g(pr_rush)); xc=shrink(c_rec,c_g,pr_rec,g(pr_rec))
    vol=shrink(c_car+c_tgt,c_g,pr_car+pr_tgt,g(pr_car)+g(pr_tgt))
    naive=shrink(c_scr,c_g,pr_scr,g(pr_scr))
    xtd=xr+xc; p_opp=1-np.exp(-xtd)
    # team expected TDs trailing
    tgn=con.sql("select season,week,game_id,posteam,team_off_td from tg").fetchnumpy()
    ts=np.asarray(tgn['season']);tw=np.asarray(tgn['week']);tt=np.asarray(tgn['posteam']);tv=np.asarray(tgn['team_off_td'],float)
    tkey=np.char.add(np.char.add(ts.astype(str),'_'),tt.astype(str))
    trail={}
    for k in np.unique(tkey):
        m=tkey==k; o=np.argsort(tw[m]); ww=tw[m][o]; vv=tv[m][o]; cs=np.cumsum(vv)
        for i in range(len(ww)): trail[(k,int(ww[i]))]=((cs[i]-vv[i])+K*2.4)/(i+K)
    team_key=np.char.add(np.char.add(season.astype(str),'_'),team.astype(str))
    team_exp=np.array([trail.get((team_key[i],int(week[i])),2.4) for i in range(len(season))])
    gt=np.char.add(np.char.add(game.astype(str),'|'),team.astype(str))
    p_coh=np.zeros(len(season))
    for k in np.unique(gt):
        m=np.where(gt==k)[0]; s=xtd[m].sum()
        if s>0: p_coh[m]=1-np.exp(-team_exp[m]*(xtd[m]/s))
    skill=np.isin(pos.astype(str),['RB','WR','TE','QB'])
    base=skill&((carries+targets)>=1)&((c_g>=1)|has_pr)&(season==test_season)
    return dict(y=scored[base],p_coh=p_coh[base],p_opp=p_opp[base],p_naive=naive[base],pos=pos[base])

# ---------- metrics ----------
def brier(y,p): return np.mean((p-y)**2)
def logloss(y,p): p=np.clip(p,1e-6,1-1e-6); return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
def auc(y,p):
    y=np.asarray(y); n1=y.sum(); n0=len(y)-n1
    if n1==0 or n0==0: return float('nan')
    order=np.argsort(p,kind='mergesort'); ranks=np.empty(len(p),float)
    sp=p[order]; i=0; r=1
    while i<len(sp):
        j=i
        while j+1<len(sp) and sp[j+1]==sp[i]: j+=1
        avg=(r+(r+(j-i)))/2.0
        ranks[order[i:j+1]]=avg; r+=(j-i+1); i=j+1
    return (ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def ece(y,p,bins=10):
    edges=np.quantile(p,np.linspace(0,1,bins+1)); edges[0]-=1e-9; edges[-1]+=1e-9; e=0.0
    for i in range(bins):
        m=(p>edges[i])&(p<=edges[i+1])
        if m.sum(): e+=m.sum()/len(p)*abs(p[m].mean()-y[m].mean())
    return e

TEST=[2022,2023,2024]
res={s:predict_season(s) for s in TEST}
pool={k:np.concatenate([res[s][k] for s in TEST]) for k in ('y','p_coh','p_opp','p_naive')}
pool_pos=np.concatenate([res[s]['pos'].astype(str) for s in TEST])

print("="*72)
print("MULTI-SEASON WALK-FORWARD BACKTEST  (each season fit on prior seasons only)")
print("="*72)
print(f"{'season':<8}{'n':>7}{'base':>8}{'Brier':>9}{'LogLoss':>9}{'AUC':>7}{'ECE':>7}{'BSS':>7}")
for s in TEST:
    y=res[s]['y']; p=res[s]['p_coh']; b=brier(y,p); bss=1-b/brier(y,np.full_like(p,y.mean()))
    print(f"{s:<8}{len(y):>7}{y.mean():>8.3f}{b:>9.4f}{logloss(y,p):>9.4f}{auc(y,p):>7.3f}{ece(y,p):>7.3f}{bss:>7.3f}")
y=pool['y']; p=pool['p_coh']; b=brier(y,p); bss=1-b/brier(y,np.full_like(p,y.mean()))
print("-"*72)
print(f"{'POOLED':<8}{len(y):>7}{y.mean():>8.3f}{b:>9.4f}{logloss(y,p):>9.4f}{auc(y,p):>7.3f}{ece(y,p):>7.3f}{bss:>7.3f}")

print("\nModel comparison (pooled 2022-24):")
print(f"{'model':<26}{'Brier':>9}{'LogLoss':>9}{'AUC':>7}")
for name,key in [('Naive: count past TDs','p_naive'),('Granular opportunity','p_opp'),('Granular + coherence','p_coh')]:
    pp=pool[key]; print(f"{name:<26}{brier(y,pp):>9.4f}{logloss(y,pp):>9.4f}{auc(y,pp):>7.3f}")

print("\nACCURACY as calibration bands (pooled, Granular+coherence):")
print(f"  {'predicted band':>16}{'n':>7}{'mean pred':>11}{'actually scored':>17}")
edges=[0,.10,.15,.20,.25,.30,.40,1.01]
for i in range(len(edges)-1):
    m=(p>=edges[i])&(p<edges[i+1])
    if m.sum(): print(f"  {int(edges[i]*100):>6}-{int(min(edges[i+1],1)*100):>3}%{m.sum():>10}{p[m].mean():>11.3f}{y[m].mean():>17.3f}")

print("\nACCURACY by tier (pooled): share of players in each tier who scored")
def tier(pr): return 'Elite (45%+)' if pr>=.45 else 'Strong (33-45)' if pr>=.33 else 'Live (22-33)' if pr>=.22 else 'Longshot (<22)'
tiers=np.array([tier(x) for x in p])
for t in ['Elite (45%+)','Strong (33-45)','Live (22-33)','Longshot (<22)']:
    m=tiers==t
    if m.sum(): print(f"  {t:<16}{m.sum():>7}  scored {y[m].mean()*100:>5.1f}%   (model avg {p[m].mean()*100:4.1f}%)")

print("\nTop-of-board precision (pooled): of the model's N highest-chance calls each pool, hit rate")
o=np.argsort(-p)
for N in (50,100,200,300):
    print(f"  top {N:<4}: {y[o[:N]].mean()*100:5.1f}% scored   (avg predicted {p[o[:N]].mean()*100:4.1f}%)")

# ---- dump results for the web board ----
import json
def band_rows(p,y):
    out=[]; edges=[0,.10,.15,.20,.25,.30,.40,1.01]
    for i in range(len(edges)-1):
        m=(p>=edges[i])&(p<edges[i+1])
        if m.sum(): out.append([int(edges[i]*100),int(min(edges[i+1],1)*100),int(m.sum()),round(float(p[m].mean()),3),round(float(y[m].mean()),3)])
    return out
tiers_out=[]
for t in ['Elite (45%+)','Strong (33-45)','Live (22-33)','Longshot (<22)']:
    m=tiers==t
    if m.sum(): tiers_out.append([t,int(m.sum()),round(float(y[m].mean()),3),round(float(p[m].mean()),3)])
dump={
 "seasons":[{"season":s,"n":int(len(res[s]['y'])),"base":round(float(res[s]['y'].mean()),3),
             "brier":round(brier(res[s]['y'],res[s]['p_coh']),4),
             "logloss":round(logloss(res[s]['y'],res[s]['p_coh']),4),
             "auc":round(auc(res[s]['y'],res[s]['p_coh']),3),
             "ece":round(ece(res[s]['y'],res[s]['p_coh']),3)} for s in TEST],
 "pooled":{"n":int(len(y)),"base":round(float(y.mean()),3),"brier":round(brier(y,p),4),
           "logloss":round(logloss(y,p),4),"auc":round(auc(y,p),3),"ece":round(ece(y,p),3),
           "bss":round(1-brier(y,p)/brier(y,np.full_like(p,y.mean())),3)},
 "models":[["Naive: count past TDs",round(brier(y,pool['p_naive']),4),round(logloss(y,pool['p_naive']),4),round(auc(y,pool['p_naive']),3)],
           ["Granular opportunity",round(brier(y,pool['p_opp']),4),round(logloss(y,pool['p_opp']),4),round(auc(y,pool['p_opp']),3)],
           ["+ Team-share coherence",round(brier(y,p),4),round(logloss(y,p),4),round(auc(y,p),3)]],
 "bands":band_rows(p,y),"tiers":tiers_out}
json.dump(dump,open('out/backtest_multi.json','w'),indent=1)
print("\nsaved out/backtest_multi.json")
