"""NFL Engine 8 stats relay.

Runs on GitHub Actions. Turns free nflverse data (play-by-play, injury reports, snap counts,
Next Gen separation, FTN charting) into one small file per season that the phone engine reads.
"""
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

VERSION = 1
START = 2012
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
REL = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DL_DIR = os.environ.get("RELAY_DL")

QB_K, QB_PRIOR, QB_DECAY, QB_USUAL_ALPHA = 150.0, -0.08, 0.96, 0.3
MISS = {"Out": 1.0, "Doubtful": 0.9}
Q_MISS = {"Did Not Participate In Practice": 0.55, "Limited Participation in Practice": 0.25,
          "Full Participation in Practice": 0.1}
POS_W = {
    "WR": 1.0, "TE": 0.6, "RB": 0.4, "HB": 0.4, "FB": 0.1, "T": 0.8, "G": 0.5, "C": 0.5, "OL": 0.6,
    "CB": 0.8, "DB": 0.6, "S": 0.6, "FS": 0.6, "SS": 0.6, "LB": 0.5, "ILB": 0.5, "MLB": 0.5, "OLB": 0.7,
    "DE": 0.8, "DT": 0.6, "NT": 0.5, "DL": 0.7,
}
OFFENSE = {"WR", "TE", "RB", "HB", "FB", "T", "G", "C", "OL", "QB"}
ALIAS = {"LV": "OAK", "OAK": "LV", "LAC": "SD", "SD": "LAC", "LA": "STL", "STL": "LA", "LAR": "LA"}
VALID = {}

EFF_COLS = ["plays", "epa_play", "pass_epa", "rush_epa", "early_down_epa", "late_down_epa", "rz_epa", "third_epa",
            "success_rate", "early_down_sr", "late_down_sr", "third_down_rate", "explosive_rate", "pass_rate",
            "sack_rate", "turnovers", "takeaways", "ppd", "rz_td_rate", "pressure_rate", "pbwr", "adj_line_yards",
            "cpoe", "sep", "drop_rate", "blitz_rate"]
OUT_COLS = ["game_id", "team"] + EFF_COLS + ["qb_id", "qb_name", "qb_val", "qb_usual", "qb_status",
                                             "inj_off", "inj_def", "inj_top"]


def log(msg):
    print(msg, flush=True)


def fetch(url):
    if DL_DIR:
        path = os.path.join(DL_DIR, url.rsplit("/", 1)[-1])
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "nfl8-relay"}), timeout=180) as r:
                data = r.read()
            if DL_DIR:
                with open(os.path.join(DL_DIR, url.rsplit("/", 1)[-1]), "wb") as f:
                    f.write(data)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except Exception as e:
            err = e
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {err}")


def csv(url, **kw):
    data = fetch(url)
    if not data:
        return None
    df = pd.read_csv(io.BytesIO(data), compression="gzip" if url.endswith(".gz") else None, low_memory=False, **kw)
    return df if len(df) else None


def team_of(t, season):
    ok = VALID.get(season, set())
    for _ in range(3):
        if t in ok or not isinstance(t, str):
            return t
        t = ALIAS.get(t, t)
    return t


def fix_team(teams, season):
    return teams.map(lambda t: team_of(t, season))


def flag(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def line_yards(y):
    y = y.fillna(0)
    return np.where(y < 0, 1.2 * y, np.where(y <= 3, y, np.where(y <= 10, 3 + 0.5 * (y - 3), 6.5)))


PBP_COLS = ["game_id", "play_id", "posteam", "defteam", "pass", "rush", "epa", "success", "down", "yardline_100",
            "yards_gained", "sack", "qb_hit", "interception", "fumble_lost", "cpoe", "qb_dropback", "qb_epa", "id",
            "rusher_player_id", "two_point_attempt", "fixed_drive", "fixed_drive_result", "pass_attempt"]


def pbp_season(season):
    p = csv(f"{REL}/pbp/play_by_play_{season}.csv.gz", usecols=lambda c: c in PBP_COLS)
    if p is None:
        return None, None, None
    p["posteam"], p["defteam"] = fix_team(p.posteam, season), fix_team(p.defteam, season)
    drives = p[p.posteam.notna() & p.fixed_drive.notna()].groupby(["game_id", "posteam", "fixed_drive"]).agg(
        res=("fixed_drive_result", "last"), minyl=("yardline_100", "min")).reset_index()
    drives["pts"] = drives.res.map({"Touchdown": 7, "Field goal": 3}).fillna(0)
    drives["td"] = (drives.res == "Touchdown").astype(float)
    drives["td_rz"] = drives.td.where(drives.minyl <= 20)
    dr = drives.groupby(["game_id", "posteam"]).agg(ppd=("pts", "mean"), rz_td_rate=("td_rz", "mean"))

    s = p[((p["pass"] == 1) | (p.rush == 1)) & p.epa.notna() & p.posteam.notna() & (p.two_point_attempt.fillna(0) == 0)].copy()
    pas, run = s["pass"] == 1, s.rush == 1
    early, late = s.down.isin([1, 2]), s.down.isin([3, 4])
    s["pass_epa"] = s.epa.where(pas)
    s["rush_epa"] = s.epa.where(run)
    s["early_down_epa"] = s.epa.where(early)
    s["late_down_epa"] = s.epa.where(late)
    s["rz_epa"] = s.epa.where(s.yardline_100 <= 20)
    s["third_epa"] = s.epa.where(s.down == 3)
    s["early_down_sr"] = s.success.where(early)
    s["late_down_sr"] = s.success.where(late)
    s["third_down_rate"] = s.success.where(s.down == 3)
    s["explosive"] = ((run & (s.yards_gained >= 10)) | (pas & (s.yards_gained >= 20))).astype(float)
    s["sack_db"] = s.sack.where(pas)
    s["hit_db"] = ((s.sack == 1) | (s.qb_hit == 1)).astype(float).where(pas)
    s["to"] = s.interception.fillna(0) + s.fumble_lost.fillna(0)
    s["line_yds"] = pd.Series(line_yards(s.yards_gained), index=s.index).where(run)
    o = s.groupby(["game_id", "posteam"]).agg(
        plays=("epa", "size"), epa_play=("epa", "mean"), pass_epa=("pass_epa", "mean"), rush_epa=("rush_epa", "mean"),
        early_down_epa=("early_down_epa", "mean"), late_down_epa=("late_down_epa", "mean"), rz_epa=("rz_epa", "mean"),
        third_epa=("third_epa", "mean"), success_rate=("success", "mean"), early_down_sr=("early_down_sr", "mean"),
        late_down_sr=("late_down_sr", "mean"), third_down_rate=("third_down_rate", "mean"),
        explosive_rate=("explosive", "mean"), pass_rate=("pass", "mean"), sack_rate=("sack_db", "mean"),
        hit_allowed=("hit_db", "mean"), turnovers=("to", "sum"), adj_line_yards=("line_yds", "mean"),
        cpoe=("cpoe", "mean")).join(dr)
    o.index.names = ["game_id", "team"]
    o = o.reset_index()
    opp = o[["game_id", "team", "hit_allowed", "turnovers"]].rename(
        columns={"team": "opponent", "hit_allowed": "pressure_rate", "turnovers": "takeaways"})
    o = o.merge(opp, on="game_id")
    o = o[o.team != o.opponent].drop(columns="opponent")
    o["pbwr"] = 1 - o.hit_allowed
    o = o.drop(columns="hit_allowed")

    qp = p[p.posteam.notna() & p.epa.notna() & (p.two_point_attempt.fillna(0) == 0)].copy()
    passers = set(qp.loc[qp.qb_dropback == 1, "id"].dropna())
    is_db = (qp.qb_dropback == 1) & qp.id.notna()
    is_run = (qp.rush == 1) & (qp.qb_dropback != 1) & qp.rusher_player_id.isin(passers)
    qp["qb_id"] = np.where(is_db, qp.id, np.where(is_run, qp.rusher_player_id, None))
    qp["val"] = np.where(is_db, qp.qb_epa.fillna(qp.epa), qp.epa)
    qb = qp[qp.qb_id.notna()].groupby(["game_id", "posteam", "qb_id"]).agg(
        plays=("val", "size"), epa_sum=("val", "sum")).reset_index().rename(columns={"posteam": "team"})
    return o, qb, p[["game_id", "play_id", "posteam", "defteam", "pass", "pass_attempt", "sack"]]


def ftn_season(season, plays):
    if season < 2022 or plays is None:
        return None
    f = csv(f"{REL}/ftn_charting/ftn_charting_{season}.csv")
    if f is None:
        return None
    f = f.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
    m = f[["game_id", "play_id", "is_drop", "n_blitzers"]].merge(plays, on=["game_id", "play_id"])
    m["drop"] = flag(m.is_drop).astype(float).where((m.pass_attempt == 1) & (m.sack != 1))
    m["blitz"] = (pd.to_numeric(m.n_blitzers, errors="coerce") > 0).astype(float).where(m["pass"] == 1)
    off = m.groupby(["game_id", "posteam"]).drop.mean().rename("drop_rate")
    dfn = m.groupby(["game_id", "defteam"]).blitz.mean().rename("blitz_rate")
    off.index.names = dfn.index.names = ["game_id", "team"]
    return pd.concat([off, dfn], axis=1).reset_index()


def ngs_separation(games):
    n = csv(f"{REL}/nextgen_stats/ngs_receiving.csv.gz")
    if n is None:
        return None
    n = n[(n.week > 0) & n.team_abbr.notna() & n.avg_separation.notna() & (n.targets > 0)].copy()
    n["team"] = [team_of(t, y) for t, y in zip(n.team_abbr, n.season)]
    n["w"] = n.avg_separation * n.targets
    t = n.groupby(["season", "week", "team"]).agg(w=("w", "sum"), n=("targets", "sum")).reset_index()
    t["sep"] = t.w / t.n
    return t.merge(team_games(games)[["season", "week", "team", "game_id"]], on=["season", "week", "team"])[
        ["game_id", "team", "sep"]]


def team_games(games):
    h = games.assign(team=games.home_team, qb_id=games.home_qb_id, qb_name=games.home_qb_name)
    a = games.assign(team=games.away_team, qb_id=games.away_qb_id, qb_name=games.away_qb_name)
    t = pd.concat([h, a])[["game_id", "season", "week", "gameday", "gametime", "team", "qb_id", "qb_name", "home_score"]]
    t["final"] = t.home_score.notna()
    return t.sort_values(["gameday", "gametime", "game_id"]).reset_index(drop=True)


def qb_ratings(games, qb_all, status):
    """Point-in-time QB value (EPA per play, shrunk toward replacement) for every team-game."""
    by_game = {k: g for k, g in qb_all.groupby(["game_id", "team"])}
    S, N, usual, seen = {}, {}, {}, {}
    rows = []

    def val(q):
        return (S.get(q, 0.0) + QB_K * QB_PRIOR) / (N.get(q, 0.0) + QB_K)

    for r in team_games(games).itertuples():
        team, q = r.team, r.qb_id if isinstance(r.qb_id, str) else None
        recent = [x for x, (season, _) in seen.get(team, {}).items() if season >= r.season - 1 and x != q]
        backup = max(recent, key=lambda x: seen[team][x][1]) if recent else None
        v = val(q) if q else usual.get(team)
        st = status.get((r.game_id, q)) if q else None
        p_miss = MISS.get(st[0], Q_MISS.get(st[1], 0.3) if st[0] == "Questionable" else 0.0) if st else 0.0
        if v is not None and p_miss:
            v = (1 - p_miss) * v + p_miss * (val(backup) if backup else QB_PRIOR)
        rows.append({"game_id": r.game_id, "team": team, "qb_id": q, "qb_name": r.qb_name if q else None,
                     "qb_val": v, "qb_usual": usual.get(team), "qb_status": st[0] if st else None})
        g = by_game.get((r.game_id, team))
        if not r.final or g is None:
            continue
        for x in g.itertuples():
            S[x.qb_id] = S.get(x.qb_id, 0.0) * QB_DECAY + x.epa_sum
            N[x.qb_id] = N.get(x.qb_id, 0.0) * QB_DECAY + x.plays
            prev = seen.setdefault(team, {}).get(x.qb_id, (r.season, 0))
            seen[team][x.qb_id] = (r.season, (prev[1] if prev[0] == r.season else 0) + x.plays)
        main = g.sort_values("plays").qb_id.iloc[-1]
        u = usual.get(team)
        usual[team] = val(main) if u is None else u + QB_USUAL_ALPHA * (val(main) - u)
    return pd.DataFrame(rows)


def injuries(season, games, players):
    inj = csv(f"{REL}/injuries/injuries_{season}.csv")
    snaps = [d for d in (csv(f"{REL}/snap_counts/snap_counts_{s}.csv") for s in (season - 1, season)) if d is not None]
    tg = team_games(games[games.season.isin([season - 1, season])])
    tg["idx"] = tg.groupby("team").cumcount()
    gidx = {(r.game_id, r.team): (r.gameday, r.idx) for r in tg.itertuples()}
    gid = {(r.season, r.week, r.team): r.game_id for r in tg.itertuples()}
    status = {}
    if inj is None:
        return pd.DataFrame(columns=["game_id", "team", "inj_off", "inj_def", "inj_top"]), status
    inj["team"] = fix_team(inj.team, season)
    inj = inj[inj.report_status.isin(["Out", "Doubtful", "Questionable"]) & inj.gsis_id.notna()]
    for r in inj.itertuples():
        g = gid.get((r.season, r.week, r.team))
        if g:
            status[(g, r.gsis_id)] = (r.report_status, r.practice_status)
    if not snaps:
        return pd.DataFrame(columns=["game_id", "team", "inj_off", "inj_def", "inj_top"]), status
    sn = pd.concat(snaps)
    sn["team"] = [team_of(t, y) for t, y in zip(sn.team, sn.season)]
    sn["gsis_id"] = sn.pfr_player_id.map(players)
    hist = {}
    for r in sn[sn.gsis_id.notna()].itertuples():
        k = gidx.get((r.game_id, r.team))
        if not k:
            continue
        off = r.position in OFFENSE
        share = r.offense_pct if off else r.defense_pct
        if share and share > 0:
            hist.setdefault(r.gsis_id, []).append((k[0], k[1], r.team, r.position, float(share)))
    for v in hist.values():
        v.sort()
    out = {}
    for r in inj.itertuples():
        g = gid.get((r.season, r.week, r.team))
        if not g or r.position == "QB":
            continue
        day, idx = gidx[(g, r.team)]
        h = [x for x in hist.get(r.gsis_id, []) if x[0] < day][-4:]
        if not h:
            continue
        pos = h[-1][3] or r.position
        missed = idx - h[-1][1] - 1 if h[-1][2] == r.team else 0
        imp = POS_W.get(pos, 0.0) * sum(x[4] for x in h) / len(h) * 0.85 ** max(missed, 0)
        p = MISS.get(r.report_status) or Q_MISS.get(r.practice_status, 0.3)
        cost = imp * p
        if cost <= 0:
            continue
        d = out.setdefault((g, r.team), {"inj_off": 0.0, "inj_def": 0.0, "top": []})
        d["inj_off" if pos in OFFENSE else "inj_def"] += cost
        d["top"].append((cost, f"{r.full_name} ({pos}, {r.report_status[0] if r.report_status != 'Doubtful' else 'D'})"))
    rows = []
    for (g, team), d in out.items():
        top = "; ".join(n for _, n in sorted(d["top"], reverse=True)[:3] if _ >= 0.25)
        rows.append({"game_id": g, "team": team, "inj_off": d["inj_off"], "inj_def": d["inj_def"], "inj_top": top or None})
    df = pd.DataFrame(rows, columns=["game_id", "team", "inj_off", "inj_def", "inj_top"])
    base = tg[tg.season == season][["game_id", "team"]].merge(df, how="left", on=["game_id", "team"])
    base[["inj_off", "inj_def"]] = base[["inj_off", "inj_def"]].fillna(0.0)
    return base, status


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:16]


def main():
    t0 = time.time()
    os.makedirs(CACHE, exist_ok=True)
    gpath = os.environ.get("GAMES_CSV")
    games = pd.read_csv(gpath) if gpath else csv(GAMES_URL)
    games = games[games.season >= START - 1]
    for season, g in games.groupby("season"):
        VALID[season] = set(g.home_team) | set(g.away_team)
    mpath = os.path.join(HERE, "manifest.json")
    man = json.load(open(mpath)) if os.path.exists(mpath) else {}
    if man.get("version") != VERSION:
        man = {"version": VERSION, "seasons": {}}
    now = datetime.now(timezone.utc)
    live = now.year if now.month >= 3 else now.year - 1
    last = int(games.season.max())
    todo = [s for s in range(START, last + 1)
            if s >= live or str(s) not in man["seasons"] or not os.path.exists(os.path.join(HERE, f"team_{s}.csv"))]
    if not todo:
        log("Nothing to rebuild")
    pl = csv(f"{REL}/players/players.csv")
    players = dict(zip(pl.pfr_id, pl.gsis_id)) if pl is not None else {}
    sep = ngs_separation(games)
    eff = {}
    for s in todo:
        log(f"{s}: play-by-play")
        o, qb, plays = pbp_season(s)
        if qb is not None:
            qb.to_csv(os.path.join(CACHE, f"qb_{s}.csv"), index=False)
        if o is not None:
            f = ftn_season(s, plays)
            if f is not None:
                o = o.merge(f, how="left", on=["game_id", "team"])
        eff[s] = o
    qb_all = pd.concat([pd.read_csv(os.path.join(CACHE, f)) for f in sorted(os.listdir(CACHE)) if f.startswith("qb_")])
    status, inj = {}, {}
    for s in todo:
        log(f"{s}: injuries and snaps")
        inj[s], st = injuries(s, games, players)
        status.update(st)
    if todo:
        log("QB ratings")
        qbr = qb_ratings(games[games.season >= START], qb_all, status)
    changed = 0
    for s in todo:
        base = team_games(games[games.season == s])[["game_id", "team"]]
        if eff[s] is not None:
            base = base.merge(eff[s], how="left", on=["game_id", "team"])
        if sep is not None:
            base = base.merge(sep, how="left", on=["game_id", "team"])
        base = base.merge(qbr, how="left", on=["game_id", "team"]).merge(inj[s], how="left", on=["game_id", "team"])
        out = base.reindex(columns=OUT_COLS)
        path = os.path.join(HERE, f"team_{s}.csv")
        out.to_csv(path, index=False, float_format="%.4f")
        h = sha(path)
        prev = man["seasons"].get(str(s), {})
        if prev.get("sha") != h:
            changed += 1
            man["seasons"][str(s)] = {"sha": h, "rows": int(len(out)), "final_rows": int(out.epa_play.notna().sum()),
                                      "updated": now.strftime("%Y-%m-%dT%H:%M:%SZ")}
            log(f"{s}: {len(out)} team-games · {int(out.epa_play.notna().sum())} with play-by-play · updated")
    stale = (now - datetime.strptime(man.get("checked", "2000-01-01T00:00:00Z"), "%Y-%m-%dT%H:%M:%SZ")
             .replace(tzinfo=timezone.utc)).days >= 7
    if changed or stale:
        man["checked"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        if changed:
            man["updated"] = man["checked"]
        with open(mpath, "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
    log(f"Done in {time.time() - t0:.0f}s · {changed} season files changed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Relay failed: {type(e).__name__}: {e}")
        sys.exit(1)
