import duckdb, numpy as np, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import pipeline as P
con=duckdb.connect()
SEASON,UPCOMING=2026,2
# effective bucket TD rates (for reallocating vacated volume to backups)
R_GL,R_RZR,R_OPEN,R_EZT,R_RZT,R_OPENT=0.38,0.08,0.006,0.42,0.09,0.015

P.build_tables(con,[2016,2017,2018,2019,2020,2021,2022,2023,2024,2025,2026],[2016,2017,2018,2019,2020,2021,2022,2023,2024,2025])
trail=P.team_env(con)

def c0(a): a=np.asarray(a,float); a[np.isnan(a)]=0.0; return a
# ---- training df (all completed rows) ----
tdf=con.sql("""select w.season,w.week,w.posteam,w.ppos,w.carries,w.targets,w.scored,
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
  left join ngs_prior np on w.pid=np.pid and w.season=np.season
  order by w.season,w.week,w.pid""").fetchnumpy()
has_pr=c0(tdf['pr_g'])>0
NGS_GM=P.ngs_global_means(tdf)
gm={n:float(np.mean(c0(tdf[n])[has_pr])) for n in ['pr_rush','pr_rec','pr_car','pr_tgt','pr_gl','pr_rzr','pr_ezt','pr_rzt','pr_scr']}
asn=np.asarray(tdf['a_snap'],float); gsnap=float(np.nanmean(asn[~np.isnan(asn)]))

def feat_rows(d, cg_key='c_g', pr_g_key='pr_g'):
    cg=c0(d[cg_key]); hp=c0(d[pr_g_key])>0
    def sh(cs,pp,gl): return (c0(cs)+P.K*np.where(hp,c0(pp),gl))/(cg+P.K)
    xr=sh(d['c_rush'],d['pr_rush'],gm['pr_rush']); xc=sh(d['c_rec'],d['pr_rec'],gm['pr_rec'])
    cpg=sh(d['c_car'],d['pr_car'],gm['pr_car']); tpg=sh(d['c_tgt'],d['pr_tgt'],gm['pr_tgt'])
    glp=sh(d['c_gl'],d['pr_gl'],gm['pr_gl']); rzr=sh(d['c_rzr'],d['pr_rzr'],gm['pr_rzr'])
    ezt=sh(d['c_ezt'],d['pr_ezt'],gm['pr_ezt']); rzt=sh(d['c_rzt'],d['pr_rzt'],gm['pr_rzt'])
    nv=sh(d['c_scr'],d['pr_scr'],gm['pr_scr'])
    asn=np.asarray(d['a_snap'],float); psn=np.asarray(d['pr_snap'],float)
    snap=np.where(~np.isnan(asn),asn,np.where(~np.isnan(psn),psn,gsnap))
    trz=c0(d['trz_tr']);tgl=c0(d['tgl_tr']);trt=c0(d['trt_tr'])
    rz_csh=np.clip((c0(d['c_gl'])+c0(d['c_rzr']))/np.maximum(trz,1e-6),0,1.2)
    gl_csh=np.clip(c0(d['c_gl'])/np.maximum(tgl,1e-6),0,1.2)
    rz_tsh=np.clip((c0(d['c_ezt'])+c0(d['c_rzt']))/np.maximum(trt,1e-6),0,1.2)
    implied=np.asarray(d['implied'],float); implied[np.isnan(implied)]=22.0
    g3=c0(d['c_g_l3'])
    def r3(cs,fb): v=c0(cs)/np.maximum(g3,1e-6); return np.where(g3>=1,v,fb)
    cpg3=r3(d['c_car_l3'],cpg); tpg3=r3(d['c_tgt_l3'],tpg); glp3=r3(d['c_gl_l3'],glp)
    a3=np.asarray(d['a_snap_l3'],float); snap3=np.where(~np.isnan(a3),a3,snap)
    trend_car=cpg3-cpg; trend_gl=glp3-glp; trend_tgt=tpg3-tpg
    out=dict(xr=xr,xc=xc,cpg=cpg,tpg=tpg,glp=glp,rzr=rzr,ezt=ezt,rzt=rzt,nv=nv,snap=snap,
                rz_csh=rz_csh,gl_csh=gl_csh,rz_tsh=rz_tsh,exp_gl_td=glp*0.38+rzr*0.08,implied=implied,cg=cg,hp=hp,
                cpg3=cpg3,tpg3=tpg3,glp3=glp3,snap3=snap3,trend_car=trend_car,trend_gl=trend_gl,trend_tgt=trend_tgt)
    out.update(P.ngs_cols(d, NGS_GM))
    return out

def stack(f, pos, team, week=None, season=None, team_exp=None):
    xtd=f['xr']+f['xc']; vol=f['cpg']+f['tpg']
    cols={'xr':f['xr'],'xc':f['xc'],'xtd':xtd,'vol':vol,'cpg':f['cpg'],'tpg':f['tpg'],'glpg':f['glp'],
          'rzrpg':f['rzr'],'eztpg':f['ezt'],'rztpg':f['rzt'],'rz_csh':f['rz_csh'],'gl_csh':f['gl_csh'],
          'rz_tsh':f['rz_tsh'],'exp_gl_td':f['exp_gl_td'],'implied':f['implied'],'team_exp':team_exp,'snap':f['snap'],'naive':f['nv'],
          'cg':f['cg'],'cpg3':f['cpg3'],'tpg3':f['tpg3'],'glp3':f['glp3'],'snap3':f['snap3'],'trend_car':f['trend_car'],'trend_gl':f['trend_gl'],'trend_tgt':f['trend_tgt'],
          **{n:f[n] for n in P.NGS},
          'is_RB':(pos=='RB'),'is_WR':(pos=='WR'),'is_TE':(pos=='TE'),'is_QB':(pos=='QB')}
    return np.column_stack([cols[k] for k in P.FEATS]).astype(float)

fT=feat_rows(tdf)
season=np.asarray(tdf['season']);week=np.asarray(tdf['week']);pos=df_pos=tdf['ppos'].astype(str);team=np.asarray(tdf['posteam'])
te_exp=np.array([trail.get((f"{int(season[i])}_{team[i]}",int(week[i])),2.4) for i in range(len(season))])
Xtr_all=stack(fT,pos,team,team_exp=te_exp)
carries=c0(tdf['carries']);targets=c0(tdf['targets']);scored=c0(tdf['scored']).astype(int)
skill=np.isin(pos,['RB','WR','TE','QB']); base=skill&((carries+targets)>=1)&((fT['cg']>=1)|fT['hp'])
train=base&((season<SEASON)|((season==SEASON)&(week<UPCOMING)))
clf=HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,
    l2_regularization=1.0,min_samples_leaf=60,random_state=0,early_stopping=False)  # raw: better calibrated for workhorses
clf.fit(Xtr_all[train],scored[train]); print("trained on",int(train.sum()))

# ---- stage-1 availability model: P(player takes >=1 offensive touch | eligible) ----
import os
INJF=[f'data/injuries_{y}.parquet' for y in range(2016,SEASON+1) if os.path.exists(f'data/injuries_{y}.parquet')]
gdf=P.build_grid(con, list(range(2016,SEASON+1)), INJF)
g_hp=c0(gdf['pr_g'])>0; g_cg=c0(gdf['c_g']); g_pos=gdf['ppos'].astype(str)
g_vol=(c0(gdf['c_car'])+c0(gdf['c_tgt']))/np.maximum(g_cg,1e-6)
X1=P.stage1_matrix(gdf, gsnap); y1=c0(gdf['played']).astype(int)
g_tr=np.isin(g_pos,['RB','WR','TE','QB'])&((g_cg>=1)|g_hp)
clf_play=HistGradientBoostingClassifier(max_depth=3,max_iter=300,learning_rate=0.05,
    l2_regularization=1.0,min_samples_leaf=60,random_state=0,early_stopping=False)
clf_play.fit(X1[g_tr],y1[g_tr]); print("availability model trained on",int(g_tr.sum()))

# ---- snapshot entering upcoming week ----
snap_df=con.sql(f"""
  with cur as (select pid,sum(gx_rush) c_rush,sum(gx_rec) c_rec,sum(carries) c_car,sum(targets) c_tgt,
      sum(n_gl) c_gl,sum(n_rzr) c_rzr,sum(n_ezt) c_ezt,sum(n_rzt) c_rzt,sum(scored) c_scr,count(*) c_g,avg(off_pct) a_snap
      from pgs where season={SEASON} and week<{UPCOMING} group by pid),
    cur3 as (select pid,sum(carries) c_car_l3,sum(targets) c_tgt_l3,sum(n_gl) c_gl_l3,sum(n_rzr) c_rzr_l3,count(*) c_g_l3,avg(off_pct) a_snap_l3
      from pgs where season={SEASON} and week<{UPCOMING} and week>={UPCOMING}-3 group by pid),
    lastteam as (select pid, arg_max(posteam,season*100+week) posteam from pg where season in ({SEASON},{SEASON-1}) group by pid),
    teamcur as (select posteam, sum(tgl) tgl_tr, sum(trz) trz_tr, sum(trt) trt_tr
      from teamrz where season={SEASON} and week<{UPCOMING} group by posteam),
    nm as (select pid, arg_max(pname,season*100+week) nm from pg where season in ({SEASON},{SEASON-1}) group by pid),
    inj as (select gsis_id pid, any_value(report_status) status, any_value(practice_status) practice from 'data/injuries_{SEASON}.parquet' where week={UPCOMING} group by gsis_id),
    curngs as (select pid, avg(sep) n_sep,avg(yacoe) n_yacoe,avg(cush) n_cush,avg(catchp) n_catchp,
                 avg(eff) n_eff,avg(ryoe) n_ryoe,avg(box8) n_box8,avg(rpoe) n_rpoe
               from ngm where season={SEASON} and week<{UPCOMING} group by pid),
    priorngs as (select * from ngs_prior where season={SEASON}),
    veg as (with g as (select home_team,away_team,total_line,spread_line from 'data/games.csv'
              where season={SEASON} and week={UPCOMING})
            select home_team team, total_line/2.0+spread_line/2.0 implied from g
            union all select away_team, total_line/2.0-spread_line/2.0 from g)
  select coalesce(cur.pid,pr.pid) pid, po.ppos, nm.nm,
    coalesce(c_rush,0) c_rush,coalesce(c_rec,0) c_rec,coalesce(c_car,0) c_car,coalesce(c_tgt,0) c_tgt,
    coalesce(c_gl,0) c_gl,coalesce(c_rzr,0) c_rzr,coalesce(c_ezt,0) c_ezt,coalesce(c_rzt,0) c_rzt,
    coalesce(c_scr,0) c_scr,coalesce(c_g,0) c_g,a_snap,
    coalesce(cur3.c_car_l3,0) c_car_l3,coalesce(cur3.c_tgt_l3,0) c_tgt_l3,coalesce(cur3.c_gl_l3,0) c_gl_l3,coalesce(cur3.c_g_l3,0) c_g_l3,cur3.a_snap_l3,
    pr.pr_rush,pr.pr_rec,pr.pr_car,pr.pr_tgt,pr.pr_gl,pr.pr_rzr,pr.pr_ezt,pr.pr_rzt,pr.pr_scr,pr.pr_snap,pr.pr_g,
    lt.posteam, tc.tgl_tr, tc.trz_tr, tc.trt_tr, veg.implied, inj.status, inj.practice,
    cn.n_sep,cn.n_yacoe,cn.n_cush,cn.n_catchp,cn.n_eff,cn.n_ryoe,cn.n_box8,cn.n_rpoe,
    pn.p_sep,pn.p_yacoe,pn.p_cush,pn.p_catchp,pn.p_eff,pn.p_ryoe,pn.p_box8,pn.p_rpoe
  from cur full outer join (select * from prior where season={SEASON}) pr on cur.pid=pr.pid
  left join cur3 on coalesce(cur.pid,pr.pid)=cur3.pid
  left join lastteam lt on coalesce(cur.pid,pr.pid)=lt.pid
  left join teamcur tc on lt.posteam=tc.posteam
  left join veg on lt.posteam=veg.team
  left join nm on coalesce(cur.pid,pr.pid)=nm.pid
  left join pos po on coalesce(cur.pid,pr.pid)=po.pid
  left join inj on coalesce(cur.pid,pr.pid)=inj.pid
  left join curngs cn on coalesce(cur.pid,pr.pid)=cn.pid
  left join priorngs pn on coalesce(cur.pid,pr.pid)=pn.pid
""").fetchnumpy()

fS=feat_rows(snap_df)
spid=np.asarray(snap_df['pid']).astype(str); spos=np.asarray(snap_df['ppos']).astype(str)
steam=np.asarray(snap_df['posteam']).astype(str); sname=np.asarray(snap_df['nm']).astype(str)
sstat=np.array([str(x) if x is not None else '' for x in snap_df['status']])
scg=fS['cg']; simplied=fS['implied']
def team_exp_now(tm):
    best=None
    for (kk,wk),v in trail.items():
        if kk==f"{SEASON}_{tm}" and (best is None or wk>best[0]): best=(wk,v)
    return best[1] if best else 2.4
steam_exp=np.array([team_exp_now(steam[i]) for i in range(len(spid))])

# schedule (unplayed upcoming games) for opponent + which teams play
sch=con.sql(f"""select away_team a, home_team h from 'data/games.csv'
  where season={SEASON} and week={UPCOMING} and result is null""").fetchall()
opp_map={}; play_set=set()
for a,h in sch: opp_map[a]=h; opp_map[h]=a; play_set.add(a); play_set.add(h)

OUT=set(['Out','Doubtful','Injured Reserve'])
# ---- injury reallocation of vacated red-zone / target volume to available teammates ----
by_team={}
for i in range(len(spid)):
    if steam[i] in play_set: by_team.setdefault(steam[i],[]).append(i)
glp=fS['glp'].copy();rzr=fS['rzr'].copy();cpg=fS['cpg'].copy()
ezt=fS['ezt'].copy();rzt=fS['rzt'].copy();tpg=fS['tpg'].copy()
xr=fS['xr'].copy();xc=fS['xc'].copy()
for tm,idxs in by_team.items():
    outs=[i for i in idxs if sstat[i] in OUT]
    if not outs: continue
    # carries -> available RBs (weight by current cpg)
    rbs=[i for i in idxs if spos[i]=='RB' and sstat[i] not in OUT]
    free_gl=sum(glp[i] for i in outs if spos[i]=='RB'); free_rzr=sum(rzr[i] for i in outs if spos[i]=='RB')
    free_open=sum(max(0,cpg[i]-glp[i]-rzr[i]) for i in outs if spos[i]=='RB')
    wsum=sum(cpg[i] for i in rbs)
    if rbs and wsum>0:
        for i in rbs:
            wgt=cpg[i]/wsum; ag=free_gl*wgt; ar=free_rzr*wgt; ao=free_open*wgt
            glp[i]+=ag; rzr[i]+=ar; cpg[i]+=ag+ar+ao
            xr[i]+=ag*R_GL+ar*R_RZR+ao*R_OPEN
    # targets -> available pass catchers (weight by current tpg)
    recs=[i for i in idxs if spos[i] in ('WR','TE','RB') and sstat[i] not in OUT]
    free_ez=sum(ezt[i] for i in outs); free_rz=sum(rzt[i] for i in outs)
    free_ot=sum(max(0,tpg[i]-ezt[i]-rzt[i]) for i in outs)
    wsum=sum(tpg[i] for i in recs)
    if recs and wsum>0:
        for i in recs:
            wgt=tpg[i]/wsum; ae=free_ez*wgt; ar=free_rz*wgt; ao=free_ot*wgt
            ezt[i]+=ae; rzt[i]+=ar; tpg[i]+=ae+ar+ao
            xc[i]+=ae*R_EZT+ar*R_RZT+ao*R_OPENT
# recompute shares from adjusted absolutes (team RZ totals scaled to per-game via c_g? use current team totals)
trz=c0(snap_df['trz_tr']);tgl=c0(snap_df['tgl_tr']);trt=c0(snap_df['trt_tr'])
fS['xr']=xr;fS['xc']=xc;fS['glp']=glp;fS['rzr']=rzr;fS['cpg']=cpg;fS['ezt']=ezt;fS['rzt']=rzt;fS['tpg']=tpg
Xs=stack(fS,spos,steam,team_exp=steam_exp)
p_conv=clf.predict_proba(Xs)[:,1]   # P(TD | player plays)
# stage-1 P(plays) for the snapshot, then combine into a true pregame probability
s1={'pr_g':snap_df['pr_g'],'c_g':snap_df['c_g'],'a_snap':snap_df['a_snap'],'a_snap_l3':snap_df['a_snap_l3'],
    'p_g_l3':snap_df['c_g_l3'],'pr_car':snap_df['pr_car'],'pr_tgt':snap_df['pr_tgt'],
    'c_car':snap_df['c_car'],'c_tgt':snap_df['c_tgt'],'status':snap_df['status'],'practice':snap_df['practice'],'ppos':snap_df['ppos']}
p_play=clf_play.predict_proba(P.stage1_matrix(s1,gsnap))[:,1]
ps=p_conv*p_play                     # anytime-TD chance = P(plays) x P(TD | plays)

skill_s=np.isin(spos,['RB','WR','TE','QB']); vol_ok=(cpg+tpg)>=3.0
rows=[]
for i in range(len(spid)):
    if not (skill_s[i] and vol_ok[i] and steam[i] in play_set and (scg[i]>=1 or fS['hp'][i])): continue
    if sstat[i] in OUT: continue   # exclude ruled-out players
    rows.append({"name":sname[i] if sname[i]!='nan' else spid[i],"team":steam[i],"opp":opp_map.get(steam[i],"—"),
        "pos":spos[i],"cpg":round(float(cpg[i]),1),"tpg":round(float(tpg[i]),1),
        "xrush":round(float(xr[i]),3),"xrec":round(float(xc[i]),3),"xtd":round(float(xr[i]+xc[i]),3),
        "rzcsh":round(float(fS['rz_csh'][i]),2),"glcsh":round(float(fS['gl_csh'][i]),2),
        "implied":round(float(simplied[i]),1),"snap":round(float(fS['snap'][i])*100,0),
        "rztsh":round(float(fS['rz_tsh'][i]),2),
        "status":sstat[i] if sstat[i] in ('Questionable',) else "",
        "avail":round(float(p_play[i]),3),"conv":round(float(p_conv[i]),4),
        "thin": bool(not fS['hp'][i]),"chance":round(float(ps[i]),4)})
rows.sort(key=lambda x:-x['chance']); rows=rows[:90]
bt=json.load(open('out/backtest_v3.json'))
asof=con.sql(f"select max(gameday) from 'data/games.csv' where season={SEASON} and week={UPCOMING} and result is null").fetchone()[0]
_rush,_rec=P.fit_rates(con,[2016,2017,2018,2019,2020,2021,2022,2023,2024,2025])
meta={"season":SEASON,"week":UPCOMING,"upcoming":True,"games":len(sch),"kickoff":str(asof),
      "rush_rates":{str(k):round(v,3) for k,v in sorted(_rush.items())},
      "rec_rates":{f'{b}_{e}':round(_rec.get((b,e),0),3) for b in range(5) for e in (0,1)},
      "out_count":int(sum(1 for i in range(len(spid)) if steam[i] in play_set and sstat[i] in OUT and skill_s[i] and (cpg[i]+tpg[i])>=3)),
      "bt":bt}
json.dump({"meta":meta,"players":rows},open('out/board_data_v3.json','w'),indent=1)
print("projected",len(rows),"players; excluded OUT/Doubtful skill:",meta['out_count'])
print("top10:",[(r['name'],r['team'],r['pos'],r['chance']) for r in rows[:10]])
