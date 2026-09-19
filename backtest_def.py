"""
Adds a walk-forward OPPONENT-DEFENSE adjustment to the model and re-runs the
multi-season backtest, comparing with vs. without the defense layer.

Defense factor per (defteam, season, up-to-week), shrunk toward 1.0:
   factor_rush = (rush TDs allowed + C) / (expected rush TDs allowed + C)
   factor_rec  = (rec  TDs allowed + C) / (expected rec  TDs allowed + C)
'Expected allowed' = sum of the opportunity model's per-play xTD faced by that D.
Applied to (a) the rush/rec mix of each player's xTD and (b) the team-TD
magnitude, then run through the same team-share coherence + Poisson.
"""
import duckdb, numpy as np
con=duckdb.connect()
def rp(s): return "read_parquet(["+",".join(f"'data/pbp_{x}.parquet'" for x in s)+"])"
ROST="read_parquet(['data/roster_2021.parquet','data/roster_2022.parquet','data/roster_2023.parquet','data/roster_2024.parquet'])"
K=3.0; C=5.0; FLO,FHI=0.6,1.6

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

def predict(test_season, use_def):
    fs=[s for s in range(2021,test_season)]; rush,rec=fit_rates(fs)
    need=[s for s in range(2021,test_season+1)]; ALL=rp(need)
    con.execute(f"""create or replace table plays as
      select season,week,game_id,posteam,defteam, rusher_player_id pid,rusher_player_name pname,
             1 isrush,1 cy,0 tg,({rushc(rush)}) rrate,0.0 crate,rush_touchdown td
        from {ALL} where rush_attempt=1 and rusher_player_id is not null and yardline_100 is not null
      union all
      select season,week,game_id,posteam,defteam, receiver_player_id,receiver_player_name,
             0,0,1,0.0,({recc(rec)}),pass_touchdown
        from {ALL} where pass_attempt=1 and receiver_player_id is not null and yardline_100 is not null""")
    con.execute("""create or replace table pg as
      select season,week,game_id,posteam,any_value(defteam) opp,pid,any_value(pname) pname,
        sum(cy) carries,sum(tg) targets,sum(rrate) gx_rush,sum(crate) gx_rec,(max(td)>0)::int scored
      from plays group by season,week,game_id,posteam,pid""")
    # defense-faced expected vs actual, per defteam-game
    con.execute("""create or replace table defg as
      select season,week,defteam,
        sum(case when isrush=1 then rrate else 0 end) exp_rush,
        sum(case when isrush=1 then td else 0 end) act_rush,
        sum(case when isrush=0 then crate else 0 end) exp_rec,
        sum(case when isrush=0 then td else 0 end) act_rec
      from plays group by season,week,defteam""")
    con.execute(f"""create or replace table pos as select gsis_id pid, any_value("position") ppos from {ROST} group by gsis_id""")
    con.execute("""create or replace table pgw as
      select p.*, po.ppos,
        coalesce(sum(gx_rush) over w,0) c_rush, coalesce(sum(gx_rec) over w,0) c_rec,
        coalesce(sum(carries) over w,0) c_car, coalesce(sum(targets) over w,0) c_tgt,
        coalesce(sum(scored) over w,0) c_scr, count(*) over w c_g
      from pg p left join pos po using(pid)
      window w as (partition by pid,season order by week rows between unbounded preceding and 1 preceding)""")
    con.execute("""create or replace table prior as
      select pid,season+1 season,avg(gx_rush) pr_rush,avg(gx_rec) pr_rec,avg(carries) pr_car,
        avg(targets) pr_tgt,avg(scored) pr_scr,count(*) pr_g from pg group by pid,season""")
    df=con.sql("""select w.season,w.week,w.game_id,w.posteam,w.opp,w.pid,w.ppos,w.carries,w.targets,w.scored,
        w.c_rush,w.c_rec,w.c_car,w.c_tgt,w.c_scr,w.c_g,
        pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_scr,pr.pr_g
      from pgw w left join prior pr on w.pid=pr.pid and w.season=pr.season""").fetchnumpy()
    def c(n): a=np.asarray(df[n],dtype=float); a[np.isnan(a)]=0.0; return a
    season=np.asarray(df['season']);week=np.asarray(df['week']);game=np.asarray(df['game_id'])
    team=np.asarray(df['posteam']);opp=np.asarray(df['opp']);pos=df['ppos']
    carries=c('carries');targets=c('targets');scored=c('scored').astype(int)
    c_rush=c('c_rush');c_rec=c('c_rec');c_car=c('c_car');c_tgt=c('c_tgt');c_scr=c('c_scr');c_g=c('c_g')
    pr_rush=c('pr_rush');pr_rec=c('pr_rec');pr_car=c('pr_car');pr_tgt=c('pr_tgt');pr_scr=c('pr_scr');has_pr=c('pr_g')>0
    g=lambda a:np.mean(a[has_pr])
    shrink=lambda cs,cn,pp,gl:(cs+K*np.where(has_pr,pp,gl))/(cn+K)
    xr=shrink(c_rush,c_g,pr_rush,g(pr_rush)); xc=shrink(c_rec,c_g,pr_rec,g(pr_rec))

    # ---- defense factors, trailing within season ----
    dg=con.sql("select season,week,defteam,exp_rush,act_rush,exp_rec,act_rec from defg").fetchnumpy()
    ds=np.asarray(dg['season']);dw=np.asarray(dg['week']);dt=np.asarray(dg['defteam'])
    er=np.asarray(dg['exp_rush'],float);ar=np.asarray(dg['act_rush'],float)
    ec=np.asarray(dg['exp_rec'],float);ac=np.asarray(dg['act_rec'],float)
    dkey=np.char.add(np.char.add(ds.astype(str),'_'),dt.astype(str))
    frush={}; frec={}
    for k in np.unique(dkey):
        m=dkey==k; o=np.argsort(dw[m])
        ww=dw[m][o]; ER=np.cumsum(er[m][o]);AR=np.cumsum(ar[m][o]);EC=np.cumsum(ec[m][o]);AC=np.cumsum(ac[m][o])
        for i in range(len(ww)):
            pe_r=ER[i]-er[m][o][i]; pa_r=AR[i]-ar[m][o][i]
            pe_c=EC[i]-ec[m][o][i]; pa_c=AC[i]-ac[m][o][i]
            frush[(k,int(ww[i]))]=min(FHI,max(FLO,(pa_r+C)/(pe_r+C)))
            frec [(k,int(ww[i]))]=min(FHI,max(FLO,(pa_c+C)/(pe_c+C)))
    if use_def:
        okey=np.char.add(np.char.add(season.astype(str),'_'),opp.astype(str))
        fr=np.array([frush.get((okey[i],int(week[i])),1.0) for i in range(len(season))])
        fc=np.array([frec .get((okey[i],int(week[i])),1.0) for i in range(len(season))])
    else:
        fr=np.ones(len(season)); fc=np.ones(len(season))
    axr=xr*fr; axc=xc*fc; adj=axr+axc
    p_opp=1-np.exp(-adj)

    # team expected TDs (offense trailing) * matchup factor, then coherent share
    tg=con.sql("""select season,week,game_id,posteam,sum(sa) t from (
        select season,week,game_id,posteam,(max(td)>0)::int sa from plays group by season,week,game_id,posteam,pid)
        group by season,week,game_id,posteam""").fetchnumpy()
    ts=np.asarray(tg['season']);tw=np.asarray(tg['week']);tt=np.asarray(tg['posteam']);tv=np.asarray(tg['t'],float)
    tkey=np.char.add(np.char.add(ts.astype(str),'_'),tt.astype(str)); trail={}
    for k in np.unique(tkey):
        m=tkey==k;o=np.argsort(tw[m]);ww=tw[m][o];vv=tv[m][o];cs=np.cumsum(vv)
        for i in range(len(ww)): trail[(k,int(ww[i]))]=((cs[i]-vv[i])+K*2.4)/(i+K)
    team_key=np.char.add(np.char.add(season.astype(str),'_'),team.astype(str))
    team_exp=np.array([trail.get((team_key[i],int(week[i])),2.4) for i in range(len(season))])
    gt=np.char.add(np.char.add(game.astype(str),'|'),team.astype(str)); p_coh=np.zeros(len(season))
    for k in np.unique(gt):
        m=np.where(gt==k)[0]; raw=(xr[m]+xc[m]).sum(); adjs=adj[m].sum()
        if adjs<=0: continue
        matchup=adjs/max(1e-9,raw)               # how this D inflates/deflates the offense overall
        te=team_exp[m]*matchup
        p_coh[m]=1-np.exp(-te*(adj[m]/adjs))
    skill=np.isin(pos.astype(str),['RB','WR','TE','QB'])
    base=skill&((carries+targets)>=1)&((c_g>=1)|has_pr)&(season==test_season)
    return dict(y=scored[base],p_coh=p_coh[base],p_opp=p_opp[base])

def brier(y,p):return np.mean((p-y)**2)
def logloss(y,p):p=np.clip(p,1e-6,1-1e-6);return -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
def auc(y,p):
    y=np.asarray(y);n1=y.sum();n0=len(y)-n1
    if n1==0 or n0==0:return float('nan')
    o=np.argsort(p,kind='mergesort');ranks=np.empty(len(p),float);sp=p[o];i=0;r=1
    while i<len(sp):
        j=i
        while j+1<len(sp) and sp[j+1]==sp[i]:j+=1
        ranks[o[i:j+1]]=(r+r+(j-i))/2.0;r+=(j-i+1);i=j+1
    return (ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def ece(y,p,b=10):
    e=np.quantile(p,np.linspace(0,1,b+1));e[0]-=1e-9;e[-1]+=1e-9;s=0.0
    for i in range(b):
        m=(p>e[i])&(p<=e[i+1])
        if m.sum():s+=m.sum()/len(p)*abs(p[m].mean()-y[m].mean())
    return s

TEST=[2022,2023,2024]
for label,ud in [("WITHOUT defense (baseline model)",False),("WITH opponent-defense adjustment",True)]:
    R={s:predict(s,ud) for s in TEST}
    y=np.concatenate([R[s]['y'] for s in TEST]); p=np.concatenate([R[s]['p_coh'] for s in TEST])
    print(f"\n{'='*64}\n{label}\n{'='*64}")
    print(f"{'season':<8}{'Brier':>9}{'LogLoss':>9}{'AUC':>7}{'ECE':>7}")
    for s in TEST:
        yy,pp=R[s]['y'],R[s]['p_coh']
        print(f"{s:<8}{brier(yy,pp):>9.4f}{logloss(yy,pp):>9.4f}{auc(yy,pp):>7.3f}{ece(yy,pp):>7.3f}")
    print(f"{'POOLED':<8}{brier(y,p):>9.4f}{logloss(y,p):>9.4f}{auc(y,p):>7.3f}{ece(y,p):>7.3f}")
