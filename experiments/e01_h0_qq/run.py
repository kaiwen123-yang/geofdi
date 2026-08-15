#!/usr/bin/env python3
"""e01 — H0 exactness suite (S1). Stages: a exactness | b correlation stress | c eps-bar & H0' | d e-process.

    python experiments/e01_h0_qq/run.py --stage a [--run-id ID] [--config path] [--quick] [--check-repro]

Figures (PNG) and tables (CSV) go to results/e01_h0_qq/<run_id>/; rollout-derived cycles to
data/processed/sim/e01_h0_qq/<run_id>/<stage>/ (both through the repo symlinks). Every stage prints a
conclusion line (PASS/FAIL + key numbers) and appends it to results/<run_id>/conclusions.txt.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one BLAS thread per worker process
    os.environ.setdefault(_v, "1")

import argparse
import copy
import datetime as _dt
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from geofdi.detect.evalue import average_run_length, eprocess, ville_frequency
from geofdi.detect.h0prime import calibrate, h0prime_test
from geofdi.detect.permutation import (
    hg_permutation_tests,
    pooled_scale,
)
from geofdi.groups.c2 import C2Rep
from geofdi.phase.registration import write_cycles
from geofdi.sim.pipeline import binom_ci, nominal_band, pmap, simulate_cycles
from geofdi.viz import figures as F

EXP_NAME = "e01_h0_qq"
REPO = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- helpers
def _sim_cfg(cfg: dict, seed: int, **over) -> dict:
    s = copy.deepcopy(cfg["sim"]); s["seed"] = int(seed); s["duration_s"] = 0.0
    for k, v in over.items():
        if k == "controller":
            s["controller"] = {**s.get("controller", {}), **v}
        else:
            s[k] = v
    return s


def _save_cycles(save_dir: Path | None, Z, meta, man):
    if save_dir is None:
        return
    write_cycles(save_dir, Z.astype(np.float32), meta, man)


def _pvals_for(Z, rep, M, seed, statistics, block_lens=(1,)):
    """{block_len: {stat: p}} using one RNG stream per block length (deterministic)."""
    out = {}
    for i, b in enumerate(block_lens):
        rng = np.random.default_rng([seed, b])
        res = hg_permutation_tests(Z, rep, statistics=statistics, M=M, rng=rng, block_len=b)
        out[b] = {s: res[s]["p"] for s in statistics}
    return out


def _size_rows(pv: np.ndarray, alphas, R, stat, label=""):
    rows = []
    for a in alphas:
        k = int(np.sum(pv <= a)); size = k / R
        lo, hi = binom_ci(k, R); blo, bhi = nominal_band(a, R)
        rows.append({"label": label, "stat": stat, "alpha": a, "size": size, "ci_lo": lo, "ci_hi": hi,
                     "band_lo": blo, "band_hi": bhi, "in_band": bool(blo <= size <= bhi), "R": R})
    return rows


def _conclude(res_dir: Path, line: str):
    print(line, flush=True)
    with open(res_dir / "conclusions.txt", "a") as fh:
        fh.write(line + "\n")


# ----------------------------------------------------------------------------- workers (top-level for fork/pickle)
def _rep_ab(sim_cfg, K, N, drop_first, M, test_seed, statistics, block_lens, save_dir):
    Z, meta, man, _ = simulate_cycles(sim_cfg, K, N=N, drop_first=drop_first)
    rep = C2Rep(man)
    _save_cycles(save_dir, Z, meta, man)
    return {"K": int(Z.shape[0]), "p": _pvals_for(Z, rep, M, test_seed, statistics, block_lens)}


def _rep_c(sim_cfg, K, N, drop_first, M, test_seed, statistics, m_list, save_dir):
    Z, meta, man, _ = simulate_cycles(sim_cfg, K, N=N, drop_first=drop_first)
    rep = C2Rep(man)
    _save_cycles(save_dir, Z, meta, man)
    out = {"K": int(Z.shape[0]), "p": {}}
    for m in m_list:
        if m > Z.shape[0]:
            continue
        out["p"][m] = _pvals_for(Z[:m], rep, M, test_seed + m, statistics)[1]
    # per-cycle antisymmetric projection summary for the eps proxy: D_k standardized, flattened mean & Gram diag
    Zs = rep.apply("s", Z); sc = pooled_scale(Z, Zs)
    D = ((Z - Zs) / sc).reshape(Z.shape[0], -1)
    out["D"] = D[::3].astype(np.float32)     # every 3rd cycle, float32 (~1.5 MB per replicate) for the eps proxy
    return out


def _rep_h0prime(sim_cfg, K_total, N, drop_first, M, test_seed, K_cal, window, block_len):
    Z, meta, man, _ = simulate_cycles(sim_cfg, K_total, N=N, drop_first=drop_first)
    rep = C2Rep(man)
    rng = np.random.default_rng(test_seed)
    Zc = Z[:K_cal]; cal = calibrate(Zc, rep, n_boot=100, block_len=block_len, rng=rng)
    ps, nus = [], []
    for w0 in range(K_cal, Z.shape[0] - window + 1, window):
        r = h0prime_test(Zc, Z[w0:w0 + window], rep, M=M, block_len=block_len, rng=rng, scale=cal["scale"])
        ps.append(r["p"]); nus.append(r["nu_mon"])
    return {"nu0": cal["nu0"], "nu0_std": cal["nu0_boot_std"], "p_windows": ps, "nu_windows": nus, "K": int(Z.shape[0])}


def _rep_d(sim_cfg, K, N, drop_first, M, test_seed, statistics, window, save_dir):
    Z, meta, man, _ = simulate_cycles(sim_cfg, K, N=N, drop_first=drop_first)
    rep = C2Rep(man)
    if save_dir is not None:
        _save_cycles(save_dir, Z[:300], {**meta, "t_start": meta["t_start"][:300], "n_cycles": 300}, man)
    out = {"K": int(Z.shape[0]), "p": {s: [] for s in statistics}}
    nwin = Z.shape[0] // window
    for w in range(nwin):
        pw = _pvals_for(Z[w * window:(w + 1) * window], rep, M, test_seed + w, statistics)[1]
        for s in statistics:
            out["p"][s].append(pw[s])
    return out


# ----------------------------------------------------------------------------- stages
def stage_a(cfg, res_dir: Path, data_dir: Path, quick=False):
    sa = cfg["stage_a"]; R = 12 if quick else sa["R"]; K = sa["K"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; M = cfg["test"]["M"]; stats_ = cfg["test"]["statistics"]
    store = "a" in cfg["outputs"]["store_cycles_stages"]
    args = [(_sim_cfg(cfg, sa["seed_base"] + r), K, N, df0, M, sa["seed_base"] + 20000 + r, stats_, (1,),
             (data_dir / "a" / f"rep_{r:03d}") if store else None) for r in range(R)]
    res = pmap(_rep_ab, args, cfg["workers"])
    P = {s: np.array([r["p"][1][s] for r in res]) for s in stats_}
    pd.DataFrame({"replicate": np.arange(R), **{f"p_{s}": P[s] for s in stats_}, "K": [r["K"] for r in res]}).to_csv(res_dir / "e01a_pvalues.csv", index=False)
    F.qq_pvalues(P, res_dir / "e01a_qq_pvalues.png", title=f"e01a — p-values under H0 (symmetric world, K={K}, M={M})")
    rows = []
    for s in stats_:
        rows += _size_rows(P[s], cfg["test"]["alphas"], R, s)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e01a_size_table.csv", index=False)
    F.size_bars(rows, res_dir / "e01a_size.png", title="e01a empirical size (bars: 95% CP CI; dashed: nominal)")
    ks = {s: stats.kstest(P[s], "uniform").pvalue for s in stats_}
    ok = all(v > 0.1 for v in ks.values()) and bool(tab["in_band"].all())
    _conclude(res_dir, f"[e01a] {'PASS' if ok else 'FAIL'}: KS p = " + ", ".join(f"{s}:{ks[s]:.3f}" for s in stats_)
              + " | sizes " + "; ".join(f"{r['stat']}@{r['alpha']}={r['size']:.3f}{'✓' if r['in_band'] else '✗'}" for r in rows)
              + f" | R={R}, K={K}, M={M}")
    return {"ks": ks, "sizes": rows, "pass": ok, "R": R}


def stage_b(cfg, res_dir: Path, data_dir: Path, quick=False):
    sb = cfg["stage_b"]; R = 12 if quick else sb["R"]; K = sb["K"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; M = cfg["test"]["M"]; stats_ = cfg["test"]["statistics"]
    block_lens = (1,) + tuple(sb["block_lens"]); alphas = cfg["test"]["alphas"]
    store = "b" in cfg["outputs"]["store_cycles_stages"]
    rows, all_p = [], []
    summary = {}
    for vname, nuis in sb["variants"].items():
        args = [(_sim_cfg(cfg, sb["seed_base"] + r, nuisance=nuis), K, N, df0, M, sb["seed_base"] + 20000 + r, stats_, block_lens,
                 (data_dir / "b" / vname / f"rep_{r:03d}") if store else None) for r in range(R)]
        res = pmap(_rep_ab, args, cfg["workers"])
        for b in block_lens:
            for s in stats_:
                pv = np.array([r["p"][b][s] for r in res])
                lab = f"{vname}|block={b}"
                rows += _size_rows(pv, alphas, R, s, label=lab)
                all_p.append(pd.DataFrame({"variant": vname, "block_len": b, "stat": s, "replicate": np.arange(R), "p": pv}))
        summary[vname] = {b: {s: float(np.mean(np.array([r["p"][b][s] for r in res]) <= cfg["test"]["alpha"])) for s in stats_} for b in block_lens}
    pd.concat(all_p).to_csv(res_dir / "e01b_pvalues.csv", index=False)
    tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e01b_size_table.csv", index=False)
    piv = tab[tab["alpha"] == cfg["test"]["alpha"]].pivot_table(index=["label"], columns="stat", values="size")
    piv.to_csv(res_dir / "e01b_size_pivot_alpha.csv")
    for vname in sb["variants"]:
        F.size_bars([r for r in rows if r["label"].startswith(vname + "|")], res_dir / f"e01b_size_{vname.replace('+','_')}.png",
                    title=f"e01b {vname}: naive (block=1) vs block flips — empirical size")
    # gate: in the stressed variant, naive inflates beyond the band at alpha and some block length is back inside
    a = cfg["test"]["alpha"]; verdicts = []
    for vname in sb["variants"]:
        naive_out = [r for r in rows if r["label"] == f"{vname}|block=1" and r["alpha"] == a and not r["in_band"] and r["size"] > r["band_hi"]]
        block_in = [r for r in rows if r["label"].startswith(vname + "|block=") and not r["label"].endswith("block=1") and r["alpha"] == a and r["in_band"]]
        verdicts.append((vname, len(naive_out), len(block_in)))
    stressed = [v for v in verdicts if v[1] > 0]
    ok = any(v[1] > 0 and v[2] > 0 for v in verdicts)
    _conclude(res_dir, f"[e01b] {'PASS' if ok else 'FAIL'}: sizes@α={a}: " + json.dumps(summary)
              + f" | naive-inflated variants: {[v[0] for v in stressed]}; verdicts (variant, #naive_out, #block_in): {verdicts}")
    return {"summary": summary, "verdicts": verdicts, "pass": ok}


def _tv_proxy(D_delta: np.ndarray, D_null: np.ndarray | None, nbins: int = 40):
    """Empirical single-cycle displacement proxy: TV between Law(s) and Law(-s), s = <D_k, w>/|w| with w the mean
    antisymmetric direction of the delta world (a data-processing LOWER bound on d_TV(Law Z, Law rho Z));
    the same estimator on null cycles is subtracted as finite-sample bias correction."""
    w = D_delta.mean(axis=0); nw = np.linalg.norm(w)
    if nw == 0:
        return 0.0, 0.0
    w = w / nw
    def tv(D):
        s = D @ w; lim = np.abs(s).max() + 1e-9; edges = np.linspace(-lim, lim, nbins + 1)
        h1, _ = np.histogram(s, edges); h2, _ = np.histogram(-s, edges)
        return 0.5 * np.abs(h1 / len(s) - h2 / len(s)).sum()
    t_delta = tv(D_delta); t_null = tv(D_null) if D_null is not None else 0.0
    return float(max(0.0, t_delta - t_null)), float(t_null)


def stage_c(cfg, res_dir: Path, data_dir: Path, quick=False):
    sc = cfg["stage_c"]; R = 8 if quick else sc["R"]; K = 60 if quick else sc["K"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; M = cfg["test"]["M"]; stats_ = cfg["test"]["statistics"]
    alpha = cfg["test"]["alpha"]; m_list = [m for m in sc["m_list"] if m <= K]
    first_n = cfg["outputs"]["store_cycles_first_n"]
    curves, bounds, rows, eps_rows = {}, {}, [], []
    D_null_pool = None
    skip_i = bool(sc.get("skip_typeI", False))        # e01c-G (Block G): only the H0' part is needed
    for delta in ([] if skip_i else [0.0] + list(sc["deltas"])):
        asym = [] if delta == 0 else [{"leg": sc["leg"], "joint": sc["joint"], "kp_gain": 1.0 + delta}]
        args = [(_sim_cfg(cfg, sc["seed_base"] + r, controller={"asymmetry": asym}), K, N, df0, M, sc["seed_base"] + 30000 + r, stats_, m_list,
                 (data_dir / "c" / f"delta_{delta}" / f"rep_{r:03d}") if r < first_n else None) for r in range(R)]
        res = pmap(_rep_c, args, cfg["workers"])
        Dpool = np.concatenate([r["D"] for r in res])                     # (R*K, dN)
        if delta == 0.0:
            D_null_pool = Dpool
            eps_hat, eps_null = 0.0, _tv_proxy(Dpool, None)[0]
        else:
            eps_hat, eps_null = _tv_proxy(Dpool, D_null_pool)
        eps_rows.append({"delta": delta, "eps_hat_proxy": eps_hat, "eps_null_estimate": eps_null, "n_cycles": Dpool.shape[0]})
        for s in stats_:
            rates, lo, hi = [], [], []
            for m in m_list:
                pv = np.array([r["p"][m][s] for r in res]); k = int(np.sum(pv <= alpha)); rate = k / R; ci = binom_ci(k, R)
                rates.append(rate); lo.append(ci[0]); hi.append(ci[1])
                rows.append({"delta": delta, "stat": s, "m": m, "rejection_rate": rate, "ci_lo": ci[0], "ci_hi": ci[1],
                             "bound_alpha_plus_m_eps": alpha + m * eps_hat, "R": R})
            curves[f"δ={delta} {s}"] = (m_list, rates, lo, hi)
        if delta > 0:
            bounds[f"bound α+m·ε̂ (δ={delta}, ε̂={eps_hat:.4f})"] = (m_list, [alpha + m * eps_hat for m in m_list])
    if not skip_i:
        tab = pd.DataFrame(rows); tab.to_csv(res_dir / "e01c_typeI_vs_m.csv", index=False)
        pd.DataFrame(eps_rows).to_csv(res_dir / "e01c_eps_proxy.csv", index=False)
        for s in stats_:
            F.typeI_vs_m({k: v for k, v in curves.items() if k.endswith(s)}, bounds, res_dir / f"e01c_typeI_vs_m_{s}.png", alpha,
                         title=f"e01c — type-I error vs monitoring window m ({s})\ndotted: coarse bound α+m·ε̂ with the empirical single-cycle proxy ε̂ (clipped at 1.05)")
        viol = tab[(tab["delta"] > 0) & (tab["rejection_rate"] > tab["bound_alpha_plus_m_eps"] + 1e-12)]
        ok_i = len(viol) == 0
    else:
        viol = []; ok_i = True
    # ---- H0'
    hp = sc["h0prime"]; Rh = 4 if quick else hp["R"]; K_cal = 40 if quick else hp["K_cal"]; K_mon = 40 if quick else hp["K_mon"]
    window = 10 if quick else hp["window"]; dbl = 60 if quick else hp["double_at_cycle"]; d0 = hp["delta"]
    period = cfg["sim"]["controller"]["period_s"]; t_double = (dbl + df0) * period
    d_change = float(hp.get("change_delta", 2 * d0))   # gain after the change (default: doubling); e01c-G: d0 = 0 (world asymmetry only)
    stable = [{"leg": sc["leg"], "joint": sc["joint"], "kp_gain": 1.0 + d0}] if d0 > 0 else []
    change = stable + [{"leg": sc["leg"], "joint": sc["joint"], "kp_gain": 1.0 + d_change, "t_start": t_double}]
    out_h = {}
    for name, asym in (("stable", stable), ("doubled", change)):
        args = [(_sim_cfg(cfg, hp["seed_base"] + r, controller={"asymmetry": asym}), K_cal + K_mon, N, df0, M,
                 hp["seed_base"] + 40000 + r, K_cal, window, hp["block_len"]) for r in range(Rh)]
        res = pmap(_rep_h0prime, args, cfg["workers"])
        Pw = np.array([r["p_windows"] for r in res])            # (Rh, n_windows)
        out_h[name] = {"P": Pw, "nu0": [r["nu0"] for r in res], "nu_windows": np.array([r["nu_windows"] for r in res])}
    Pst = out_h["stable"]["P"]; Pdb = out_h["doubled"]["P"]
    nwin = Pst.shape[1]; change_win = max(0, (dbl - K_cal) // window)
    size_h = float(np.mean(Pst <= alpha)); k = int(np.sum(Pst <= alpha)); ci = binom_ci(k, Pst.size); band = nominal_band(alpha, Pst.size)
    pre = Pdb[:, :change_win]; post = Pdb[:, change_win:]
    power_post = float(np.mean(post <= alpha)) if post.size else float("nan"); size_pre = float(np.mean(pre <= alpha)) if pre.size else float("nan")
    alarm_frac = float(np.mean((post <= alpha).any(axis=1))) if post.size else float("nan")
    pd.DataFrame({"replicate": np.repeat(np.arange(Rh), nwin), "window": np.tile(np.arange(nwin), Rh),
                  "p_stable": Pst.ravel(), "p_doubled": Pdb.ravel(), "nu_stable": out_h["stable"]["nu_windows"].ravel(),
                  "nu_doubled": out_h["doubled"]["nu_windows"].ravel()}).to_csv(res_dir / "e01c_h0prime_windows.csv", index=False)
    pd.DataFrame([{"delta": d0, "R": Rh, "K_cal": K_cal, "K_mon": K_mon, "window": window, "n_windows": nwin,
                   "size_stable": size_h, "ci_lo": ci[0], "ci_hi": ci[1], "band_lo": band[0], "band_hi": band[1],
                   "nu0_mean": float(np.mean(out_h["stable"]["nu0"])), "change_window": change_win, "size_pre_change": size_pre,
                   "power_post_change": power_post, "alarm_fraction_post": alarm_frac}]).to_csv(res_dir / "e01c_h0prime_summary.csv", index=False)
    F.h0prime_timeline(np.median(Pdb, axis=0), change_win, alpha, res_dir / "e01c_h0prime_timeline.png",
                       title=f"e01c — H0′ p-values per monitoring window (median over {Rh} runs); δ={d0} doubles at window {change_win}")
    ok_h = (band[0] <= size_h <= band[1] or ci[0] <= alpha <= ci[1]) and alarm_frac >= 0.8
    ok = ok_i and ok_h
    _conclude(res_dir, f"[e01c] {'PASS' if ok else 'FAIL'}: (i) type-I ≤ α+m·ε̂ everywhere: {ok_i} ({len(viol)} violations; ε̂ proxy = "
              + ", ".join(f"δ={r['delta']}:{r['eps_hat_proxy']:.4f}" for r in eps_rows if r['delta'] > 0)
              + f") | (ii) H0′ size(stable)={size_h:.3f} [band {band[0]:.3f},{band[1]:.3f}], alarm fraction after δ doubling={alarm_frac:.2f}, "
              + f"post-change window rejection={power_post:.2f} (pre {size_pre:.2f})")
    return {"pass": ok, "pass_i": ok_i, "pass_h0prime": ok_h}


def stage_d(cfg, res_dir: Path, data_dir: Path, quick=False):
    sd = cfg["stage_d"]; S = 4 if quick else sd["seeds"]; K = 300 if quick else sd["K"]; window = sd["window"]
    N = cfg["registration"]["N"]; df0 = cfg["registration"]["drop_first"]; M = cfg["test"]["M"]; stats_ = cfg["test"]["statistics"]
    alpha = cfg["test"]["alpha"]; first_n = cfg["outputs"]["store_cycles_first_n"]
    args = [(_sim_cfg(cfg, sd["seed_base"] + s), K, N, df0, M, sd["seed_base"] + 50000 + s, stats_, window,
             (data_dir / "d" / f"seed_{s:03d}") if s < first_n else None) for s in range(S)]
    res = pmap(_rep_d, args, cfg["workers"])
    rows, Es = [], {s: [] for s in stats_}
    for st in stats_:
        alarms = []
        for i, r in enumerate(res):
            E, al = eprocess(r["p"][st], alpha=alpha); Es[st].append(E); alarms.append(al)
            rows.append({"stat": st, "seed": i, "n_windows": len(E), "max_E": float(E.max()), "alarm_window": -1 if al is None else al})
        vf = ville_frequency(alarms); arl = average_run_length(alarms, len(res[0]["p"][st]))
        pd.DataFrame(rows).to_csv(res_dir / "e01d_alarms.csv", index=False)
        F.eprocess_trajectories(Es[st], alpha, res_dir / f"e01d_eprocess_{st}.png",
                                title=f"e01d — running e-values on nominal streams ({st}; {S} seeds × {K} cycles, window {window})", window_cycles=window)
    summ = []
    for st in stats_:
        alarms = [None if r == -1 else r for r in pd.DataFrame(rows).query("stat == @st")["alarm_window"]]
        vf = ville_frequency(alarms); arl = average_run_length(alarms, len(res[0]["p"][st]))
        summ.append({"stat": st, "seeds": S, "cycles_per_seed": K, "windows_per_seed": len(res[0]["p"][st]), "window": window, "alpha": alpha,
                     "ville_frequency": vf, "ville_bound": alpha, "arl_censored_windows": arl["arl_censored_mean"],
                     "arl_censored_cycles": arl["arl_censored_mean"] * window, "n_alarms": arl["n_alarms"], "one_over_alpha": 1 / alpha})
    st_df = pd.DataFrame(summ); st_df.to_csv(res_dir / "e01d_ville_arl.csv", index=False)
    ok = bool((st_df["ville_frequency"] <= alpha).all() and (st_df["arl_censored_windows"] >= 0.8 / alpha).all())
    _conclude(res_dir, f"[e01d] {'PASS' if ok else 'FAIL'}: " + "; ".join(f"{r['stat']}: Ville freq={r['ville_frequency']:.3f} (≤α={alpha}), "
              f"ARL={r['arl_censored_windows']:.1f} windows (={r['arl_censored_cycles']:.0f} cycles; ≥0.8/α={0.8/alpha:.0f}), alarms={r['n_alarms']}/{S}" for r in summ))
    return {"pass": ok, "summary": summ}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["a", "b", "c", "d", "all"], default="all")
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--run-id", default=_dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--quick", action="store_true", help="tiny replicate counts (smoke test)")
    ap.add_argument("--check-repro", action="store_true", help="rerun the stage into <run>/repro and byte-compare the CSVs")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.workers:
        cfg["workers"] = args.workers
    res_dir = REPO / "results" / EXP_NAME / args.run_id; res_dir.mkdir(parents=True, exist_ok=True)
    data_dir = REPO / "data" / "processed" / "sim" / EXP_NAME / args.run_id; data_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    stages = ["a", "b", "c", "d"] if args.stage == "all" else [args.stage]
    print(f"[{EXP_NAME}] run_id={args.run_id} stages={stages} quick={args.quick} results={res_dir}", flush=True)
    fn = {"a": stage_a, "b": stage_b, "c": stage_c, "d": stage_d}
    for s in stages:
        t0 = _dt.datetime.now()
        fn[s](cfg, res_dir, data_dir, quick=args.quick)
        print(f"  stage {s} done in {(_dt.datetime.now() - t0).total_seconds():.0f}s", flush=True)
        if args.check_repro:
            rdir = res_dir / "repro" / s; rdir.mkdir(parents=True, exist_ok=True)
            ddir = data_dir / "repro" / s
            fn[s](cfg, rdir, ddir, quick=args.quick)
            same = []
            for f in sorted(res_dir.glob(f"e01{s}_*.csv")):
                g = rdir / f.name
                same.append((f.name, g.exists() and f.read_bytes() == g.read_bytes()))
            _conclude(res_dir, f"[repro {s}] " + ("PASS" if all(v for _, v in same) else "FAIL") + " byte-identical CSVs: " + str(same))
            shutil.rmtree(ddir, ignore_errors=True)


if __name__ == "__main__":
    main()
