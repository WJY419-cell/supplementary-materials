"""Experiments for a concise cross-layer coupled rank aggregation model.

The implementation intentionally contains no source-reliability estimation,
malicious-source model, label-flip likelihood, calibration/freezing stage, or
feedback from the explanation module.  The core model is

  delta_ijk = (u_i-u_j) + (B_ik-B_jk)

with ridge penalties and a single graph-Laplacian coupling penalty on B.

Real-data protocols use natural contexts or typed user groups as layers:
Breakfast contexts, SUSHI geographic-region meta-paths, and MovieLens
occupation meta-paths.  Raw SUSHI and MovieLens files retain their upstream
licenses and must not be redistributed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib-cache") if "ROOT" in globals() else str(Path.cwd() / "tmp/matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from scipy.stats import kendalltau, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
# Portable data locations; no changes to model or experimental calculations.
DATA_ROOT = Path(os.environ.get("MPC_DATA_ROOT", str(ROOT / "data"))).expanduser()
BREAKFAST = DATA_ROOT / "breakfast/00035-00000001.csv"
SUSHI_DIR = DATA_ROOT / "sushi3-2016"
MOVIELENS_DIR = DATA_ROOT / "ml-100k"


@dataclass
class Record:
    unit: int
    layer: int
    items: np.ndarray
    values: np.ndarray


@dataclass
class Dataset:
    name: str
    records: list[Record]
    n_items: int
    layer_names: list[str]
    similarity: np.ndarray
    aligned_units: bool = False
    test_fraction: float = 0.25
    train_cap_per_layer: int | None = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def center(u: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = u - u.mean()
    if b.size:
        b = b - b.mean(axis=0, keepdims=True)
        common = b.mean(axis=1, keepdims=True)
        b = b - common
        u = u + common[:, 0]
        u = u - u.mean()
    return u, b


def counts_from_records(records: list[Record], n_items: int, n_layers: int):
    counts: dict[tuple[int, int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for rec in records:
        m = len(rec.items)
        for a in range(m):
            for z in range(a + 1, m):
                i, j = int(rec.items[a]), int(rec.items[z])
                vi, vj = float(rec.values[a]), float(rec.values[z])
                if vi == vj or i == j:
                    continue
                lo, hi = (i, j) if i < j else (j, i)
                i_wins = vi > vj
                if i > j:
                    i_wins = not i_wins
                counts[(lo, hi, rec.layer)][0 if i_wins else 1] += 1.0
    keys = sorted(counts)
    if not keys:
        raise ValueError("No pairwise comparisons were constructed")
    i = np.array([x[0] for x in keys], dtype=np.int32)
    j = np.array([x[1] for x in keys], dtype=np.int32)
    k = np.array([x[2] for x in keys], dtype=np.int32)
    win = np.array([counts[x][0] for x in keys], dtype=float)
    lose = np.array([counts[x][1] for x in keys], dtype=float)
    return i, j, k, win, lose


def laplacian(similarity: np.ndarray) -> np.ndarray:
    s = np.asarray(similarity, dtype=float).copy()
    np.fill_diagonal(s, 0.0)
    if np.max(s) > 0:
        s /= np.max(s)
    d = s.sum(axis=1)
    return np.diag(d) - s


def fit_joint(
    counts,
    n_items: int,
    n_layers: int,
    similarity: np.ndarray,
    lambda_u: float = 0.03,
    lambda_b: float = 0.08,
    lambda_s: float = 0.8,
    steps: int = 900,
    lr: float = 0.08,
):
    i, j, k, win, lose = counts
    total = max(float((win + lose).sum()), 1.0)
    u = np.zeros(n_items, dtype=float)
    b = np.zeros((n_items, n_layers), dtype=float)
    mu = np.zeros_like(u)
    vu = np.zeros_like(u)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    L = laplacian(similarity)
    beta1, beta2 = 0.9, 0.999
    for t in range(1, steps + 1):
        delta = (u[i] - u[j]) + (b[i, k] - b[j, k])
        g = ((win + lose) * sigmoid(delta) - win) / total
        gu = lambda_u * u
        gb = lambda_b * b + lambda_s * (b @ L) / max(n_layers, 1)
        np.add.at(gu, i, g)
        np.add.at(gu, j, -g)
        np.add.at(gb, (i, k), g)
        np.add.at(gb, (j, k), -g)
        mu = beta1 * mu + (1 - beta1) * gu
        vu = beta2 * vu + (1 - beta2) * (gu * gu)
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * (gb * gb)
        ahat = mu / (1 - beta1**t)
        vhat = vu / (1 - beta2**t)
        bhat = mb / (1 - beta1**t)
        bvhat = vb / (1 - beta2**t)
        rate = lr * (0.35 + 0.65 * (1 - t / (steps + 1)))
        u -= rate * ahat / (np.sqrt(vhat) + 1e-8)
        b -= rate * bhat / (np.sqrt(bvhat) + 1e-8)
        u, b = center(u, b)
    return u, b


def fit_pooled(counts, n_items: int, n_layers: int, steps: int = 800):
    ones = np.ones((n_layers, n_layers), dtype=float)
    u, _ = fit_joint(counts, n_items, n_layers, ones, lambda_b=1e6, lambda_s=0.0, steps=steps)
    return u


def fit_layerwise(counts, n_items: int, n_layers: int, steps: int = 800):
    """Fit one Bradley--Terry model per layer, without cross-layer sharing.

    This is an application baseline rather than an ablation of MPC-Rank: it
    represents the common practice of maintaining a separate ranking model for
    each context or audience segment.
    """
    i, j, k, win, lose = counts
    offsets = np.zeros((n_items, n_layers), dtype=float)
    for layer in range(n_layers):
        mask = k == layer
        if not np.any(mask):
            continue
        local = (i[mask], j[mask], np.zeros(int(mask.sum()), dtype=np.int32),
                 win[mask], lose[mask])
        offsets[:, layer] = fit_pooled(local, n_items, 1, steps=steps)
    return np.zeros(n_items, dtype=float), offsets


def borda_scores(counts, n_items: int) -> np.ndarray:
    i, j, _k, win, lose = counts
    score = np.zeros(n_items, dtype=float)
    np.add.at(score, i, win - lose)
    np.add.at(score, j, lose - win)
    return score - score.mean()


def calibrate_scale(counts, score: np.ndarray, offsets: np.ndarray | None = None):
    """Choose a positive temperature on training data without changing ranks."""
    i, j, k, win, lose = counts
    raw = score[i] - score[j]
    if offsets is not None:
        raw += offsets[i, k] - offsets[j, k]
    best_alpha, best_loss = 1.0, float("inf")
    total = max(float((win + lose).sum()), 1.0)
    for alpha in np.logspace(-3, 1.5, 80):
        p = sigmoid(alpha * raw)
        loss = -(win * np.log(p + 1e-12) + lose * np.log(1 - p + 1e-12)).sum() / total
        if loss < best_loss:
            best_alpha, best_loss = float(alpha), float(loss)
    return score * best_alpha, None if offsets is None else offsets * best_alpha


def rank_centrality(counts, n_items: int, iterations: int = 300) -> np.ndarray:
    i, j, _k, win, lose = counts
    w = np.zeros((n_items, n_items), dtype=float)
    np.add.at(w, (i, j), win)
    np.add.at(w, (j, i), lose)
    games = w + w.T
    deg = np.maximum((games > 0).sum(axis=1).max(), 1)
    p = np.zeros_like(w)
    nz = games > 0
    p[nz] = w.T[nz] / games[nz]
    p /= deg
    np.fill_diagonal(p, 1.0 - p.sum(axis=1))
    q = np.full(n_items, 1.0 / n_items)
    for _ in range(iterations):
        q = q @ p
        q = np.maximum(q, 1e-14)
        q /= q.sum()
    return np.log(q) - np.log(q).mean()


def split_records(ds: Dataset, seed: int, test_fraction: float = 0.25):
    rng = np.random.RandomState(seed)
    if ds.aligned_units:
        units = np.array(sorted({r.unit for r in ds.records}))
        rng.shuffle(units)
        nt = max(2, int(round(test_fraction * len(units))))
        test_units = set(units[:nt].tolist())
    else:
        test_units = set()
        for layer in range(len(ds.layer_names)):
            units = np.array(sorted({r.unit for r in ds.records if r.layer == layer}))
            rng.shuffle(units)
            nt = max(1, int(round(test_fraction * len(units))))
            test_units.update(units[:nt].tolist())
    train = [r for r in ds.records if r.unit not in test_units]
    test = [r for r in ds.records if r.unit in test_units]
    return train, test


def pair_metrics(counts, global_scores: np.ndarray, layer_offsets: np.ndarray | None):
    i, j, k, win, lose = counts
    delta = global_scores[i] - global_scores[j]
    if layer_offsets is not None:
        delta += layer_offsets[i, k] - layer_offsets[j, k]
    p = sigmoid(delta)
    total = max(float((win + lose).sum()), 1.0)
    acc = (win * (p >= 0.5) + lose * (p < 0.5)).sum() / total
    nll = -(win * np.log(p + 1e-12) + lose * np.log(1 - p + 1e-12)).sum() / total
    return float(acc), float(nll)


def consensus_scores(counts, n_items: int) -> np.ndarray:
    return borda_scores(counts, n_items)


def tau_score(pred: np.ndarray, truth: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(truth)
    val = kendalltau(pred[mask], truth[mask]).correlation
    return float(0.0 if np.isnan(val) else val)


def evaluate_dataset(ds: Dataset, seeds: list[int], steps: int):
    rows = []
    n_layers = len(ds.layer_names)
    for seed in seeds:
        train, test = split_records(ds, seed, test_fraction=ds.test_fraction)
        if ds.train_cap_per_layer is not None:
            rng = np.random.RandomState(seed + 731)
            capped = []
            for layer in range(n_layers):
                part = [r for r in train if r.layer == layer]
                rng.shuffle(part)
                capped.extend(part[: ds.train_cap_per_layer])
            train = capped
        tr = counts_from_records(train, ds.n_items, n_layers)
        te = counts_from_records(test, ds.n_items, n_layers)
        truth = consensus_scores(te, ds.n_items)
        methods = {}
        methods["Borda"] = calibrate_scale(tr, borda_scores(tr, ds.n_items))
        methods["RankCentrality"] = calibrate_scale(tr, rank_centrality(tr, ds.n_items))
        methods["Pooled-BT"] = (fit_pooled(tr, ds.n_items, n_layers, steps=max(500, steps - 100)), None)
        # Coupling strength is selected only from an inner validation split.
        inner_ds = Dataset(ds.name, train, ds.n_items, ds.layer_names, ds.similarity,
                           ds.aligned_units, 0.2, None)
        inner_train, inner_val = split_records(inner_ds, seed + 10007, test_fraction=0.2)
        inner_tr = counts_from_records(inner_train, ds.n_items, n_layers)
        inner_va = counts_from_records(inner_val, ds.n_items, n_layers)
        best_lambda, best_nll = 0.05, float("inf")
        for candidate in (0.01, 0.05, 0.2, 0.8):
            cu, cb = fit_joint(inner_tr, ds.n_items, n_layers, ds.similarity,
                               lambda_u=0.03, lambda_b=0.005, lambda_s=candidate,
                               steps=max(180, steps // 2))
            _acc, val_nll = pair_metrics(inner_va, cu, cb)
            if val_nll < best_nll:
                best_lambda, best_nll = candidate, val_nll
        u, b = fit_joint(tr, ds.n_items, n_layers, ds.similarity,
                         lambda_u=0.03, lambda_b=0.005, lambda_s=best_lambda, steps=steps)
        methods["MPC-Rank"] = (u, b)
        for method, (score, offsets) in methods.items():
            acc, nll = pair_metrics(te, score, offsets)
            rows.append({
                "dataset": ds.name,
                "seed": seed,
                "method": method,
                "kendall_tau": tau_score(score, truth),
                "pair_accuracy": acc,
                "test_nll": nll,
                "train_records": len(train),
                "test_records": len(test),
                "n_items": ds.n_items,
                "n_layers": n_layers,
                "selected_lambda_s": best_lambda if method == "MPC-Rank" else "",
            })
    return rows


def cap_training_records(records: list[Record], n_layers: int, cap: int, seed: int):
    """Take at most ``cap`` ranking units from every real-data layer."""
    rng = np.random.RandomState(seed + 731)
    capped = []
    for layer in range(n_layers):
        part = [r for r in records if r.layer == layer]
        rng.shuffle(part)
        capped.extend(part[:cap])
    return capped


def select_coupling(ds: Dataset, train: list[Record], seed: int, steps: int):
    """Select lambda_s using only an inner split of the currently available data."""
    n_layers = len(ds.layer_names)
    inner_ds = Dataset(ds.name, train, ds.n_items, ds.layer_names, ds.similarity,
                       ds.aligned_units, 0.2, None)
    inner_train, inner_val = split_records(inner_ds, seed + 10007, test_fraction=0.2)
    inner_tr = counts_from_records(inner_train, ds.n_items, n_layers)
    inner_va = counts_from_records(inner_val, ds.n_items, n_layers)
    best_lambda, best_nll = 0.05, float("inf")
    for candidate in (0.01, 0.05, 0.2, 0.8):
        cu, cb = fit_joint(inner_tr, ds.n_items, n_layers, ds.similarity,
                           lambda_u=0.03, lambda_b=0.005, lambda_s=candidate,
                           steps=max(160, steps // 2))
        _acc, val_nll = pair_metrics(inner_va, cu, cb)
        if val_nll < best_nll:
            best_lambda, best_nll = candidate, val_nll
    return best_lambda


def evaluate_support_sweep(ds: Dataset, seeds: list[int], caps: list[int], steps: int):
    """Evaluate real-data prediction as the number of records per layer decreases."""
    rows = []
    n_layers = len(ds.layer_names)
    for seed in seeds:
        full_train, test = split_records(ds, seed, test_fraction=ds.test_fraction)
        te = counts_from_records(test, ds.n_items, n_layers)
        truth = consensus_scores(te, ds.n_items)
        for cap in caps:
            train = cap_training_records(full_train, n_layers, cap, seed + 1000 * cap)
            tr = counts_from_records(train, ds.n_items, n_layers)
            pooled = (fit_pooled(tr, ds.n_items, n_layers, steps=max(420, steps - 100)), None)
            separate = fit_layerwise(tr, ds.n_items, n_layers, steps=max(360, steps - 150))
            best_lambda = select_coupling(ds, train, seed + 1000 * cap, steps)
            joint = fit_joint(tr, ds.n_items, n_layers, ds.similarity,
                              lambda_u=0.03, lambda_b=0.005,
                              lambda_s=best_lambda, steps=steps)
            for method, (score, offsets) in {
                "Pooled-BT": pooled,
                "Layerwise-BT": separate,
                "MPC-Rank": joint,
            }.items():
                acc, nll = pair_metrics(te, score, offsets)
                rows.append({
                    "dataset": ds.name,
                    "seed": seed,
                    "records_per_layer": cap,
                    "method": method,
                    "kendall_tau": "" if method == "Layerwise-BT" else tau_score(score, truth),
                    "pair_accuracy": acc,
                    "test_nll": nll,
                    "train_records": len(train),
                    "test_records": len(test),
                    "selected_lambda_s": best_lambda if method == "MPC-Rank" else "",
                })
    return rows


def summarize_support(rows):
    out = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], int(row["records_per_layer"]), row["method"])].append(row)
    for (dataset, cap, method), vals in sorted(groups.items()):
        entry = {"dataset": dataset, "records_per_layer": cap,
                 "method": method, "n_runs": len(vals)}
        for metric in ("pair_accuracy", "test_nll"):
            x = np.array([float(v[metric]) for v in vals])
            entry[f"{metric}_mean"] = float(x.mean())
            entry[f"{metric}_std"] = float(x.std(ddof=1))
            entry[f"{metric}_ci95"] = float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))
        out.append(entry)
    return out


def support_significance(rows):
    """Paired one-sided tests for MPC-Rank at every real-data support level."""
    out = []
    keys = sorted({(r["dataset"], int(r["records_per_layer"])) for r in rows})
    for dataset, cap in keys:
        for metric in ("pair_accuracy", "test_nll"):
            proposed = {int(r["seed"]): float(r[metric]) for r in rows
                        if r["dataset"] == dataset and int(r["records_per_layer"]) == cap
                        and r["method"] == "MPC-Rank"}
            for baseline in ("Pooled-BT", "Layerwise-BT"):
                base = {int(r["seed"]): float(r[metric]) for r in rows
                        if r["dataset"] == dataset and int(r["records_per_layer"]) == cap
                        and r["method"] == baseline}
                common = sorted(set(proposed) & set(base))
                diff = np.array([(base[s] - proposed[s]) if metric == "test_nll"
                                 else (proposed[s] - base[s]) for s in common])
                try:
                    p = float(wilcoxon(diff, alternative="greater").pvalue)
                except ValueError:
                    p = 1.0
                out.append({"dataset": dataset, "records_per_layer": cap,
                            "metric": metric, "baseline": baseline, "n": len(common),
                            "mean_improvement": float(diff.mean()),
                            "wilcoxon_one_sided_p": p})
    return out


def evaluate_long_tail_layers(ds: Dataset, seeds: list[int], supports: list[int], steps: int):
    """Evaluate sparse target layers while the remaining relation layers stay data-rich.

    For every outer split, a quarter of the layers are selected before model fitting by a
    seed-controlled permutation.  Only their training records are restricted; evaluation
    is performed exclusively on held-out records from those target layers.  This emulates
    newly introduced or long-tail relation layers without modifying test labels.
    """
    rows = []
    n_layers = len(ds.layer_names)
    n_target = max(1, int(math.ceil(n_layers / 4)))
    for seed in seeds:
        full_train, test = split_records(ds, seed, test_fraction=ds.test_fraction)
        rng = np.random.RandomState(seed + 4242)
        target_layers = sorted(rng.permutation(n_layers)[:n_target].tolist())
        target_set = set(target_layers)
        target_test = [r for r in test if r.layer in target_set]
        te = counts_from_records(target_test, ds.n_items, n_layers)

        # Keep donor layers at the same maximum support used in the main real-data setting.
        donors = []
        for layer in range(n_layers):
            if layer in target_set:
                continue
            part = [r for r in full_train if r.layer == layer]
            rng.shuffle(part)
            if ds.train_cap_per_layer is not None:
                part = part[:ds.train_cap_per_layer]
            donors.extend(part)

        for support in supports:
            train = list(donors)
            for layer in target_layers:
                part = [r for r in full_train if r.layer == layer]
                rng_layer = np.random.RandomState(seed * 10007 + support * 101 + layer)
                rng_layer.shuffle(part)
                train.extend(part[:support])
            tr = counts_from_records(train, ds.n_items, n_layers)
            pooled = (fit_pooled(tr, ds.n_items, n_layers, steps=max(420, steps - 100)), None)
            separate = fit_layerwise(tr, ds.n_items, n_layers, steps=max(360, steps - 150))
            best_lambda = select_coupling(ds, train, seed + 5000 * (support + 1), steps)
            joint = fit_joint(tr, ds.n_items, n_layers, ds.similarity,
                              lambda_u=0.03, lambda_b=0.005,
                              lambda_s=best_lambda, steps=steps)
            for method, (score, offsets) in {
                "Pooled-BT": pooled,
                "Layerwise-BT": separate,
                "MPC-Rank": joint,
            }.items():
                acc, nll = pair_metrics(te, score, offsets)
                rows.append({
                    "dataset": ds.name,
                    "seed": seed,
                    "target_support": support,
                    "method": method,
                    "pair_accuracy": acc,
                    "test_nll": nll,
                    "target_layers": "|".join(str(x) for x in target_layers),
                    "n_target_layers": n_target,
                    "train_records": len(train),
                    "target_test_records": len(target_test),
                    "selected_lambda_s": best_lambda if method == "MPC-Rank" else "",
                })
    return rows


def summarize_long_tail(rows):
    out = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], int(row["target_support"]), row["method"])].append(row)
    for (dataset, support, method), vals in sorted(groups.items()):
        entry = {"dataset": dataset, "target_support": support,
                 "method": method, "n_runs": len(vals)}
        for metric in ("pair_accuracy", "test_nll"):
            x = np.array([float(v[metric]) for v in vals])
            entry[f"{metric}_mean"] = float(x.mean())
            entry[f"{metric}_std"] = float(x.std(ddof=1))
            entry[f"{metric}_ci95"] = float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))
        out.append(entry)
    return out


def long_tail_significance(rows):
    out = []
    keys = sorted({(r["dataset"], int(r["target_support"])) for r in rows})
    for dataset, support in keys:
        for metric in ("pair_accuracy", "test_nll"):
            proposed = {int(r["seed"]): float(r[metric]) for r in rows
                        if r["dataset"] == dataset and int(r["target_support"]) == support
                        and r["method"] == "MPC-Rank"}
            for baseline in ("Pooled-BT", "Layerwise-BT"):
                base = {int(r["seed"]): float(r[metric]) for r in rows
                        if r["dataset"] == dataset and int(r["target_support"]) == support
                        and r["method"] == baseline}
                common = sorted(set(proposed) & set(base))
                diff = np.array([(base[s] - proposed[s]) if metric == "test_nll"
                                 else (proposed[s] - base[s]) for s in common])
                try:
                    p = float(wilcoxon(diff, alternative="greater").pvalue)
                except ValueError:
                    p = 1.0
                out.append({"dataset": dataset, "target_support": support,
                            "metric": metric, "baseline": baseline, "n": len(common),
                            "mean_improvement": float(diff.mean()),
                            "wilcoxon_one_sided_p": p})
    return out


def plot_long_tail(summary_rows, out: Path):
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=str(font_path)).get_name()]
        plt.rcParams["axes.unicode_minus"] = False
    datasets = ["Breakfast-6Context", "SUSHI-RegionPaths", "MovieLens-OccupationPaths"]
    methods = ["Pooled-BT", "Layerwise-BT", "MPC-Rank"]
    display = {"Pooled-BT": "汇聚 BT", "Layerwise-BT": "逐层 BT", "MPC-Rank": "MPC-Rank"}
    colors = {"Pooled-BT": "#8d99a8", "Layerwise-BT": "#d08b45", "MPC-Rank": "#174f83"}
    markers = {"Pooled-BT": "s", "Layerwise-BT": "^", "MPC-Rank": "o"}
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.45))
    for col, dataset in enumerate(datasets):
        ax = axes[col]
        subset = [r for r in summary_rows if r["dataset"] == dataset]
        for method in methods:
            vals = sorted((r for r in subset if r["method"] == method),
                          key=lambda r: int(r["target_support"]))
            x = np.array([int(r["target_support"]) for r in vals])
            y = np.array([float(r["test_nll_mean"]) for r in vals])
            e = np.array([float(r["test_nll_ci95"]) for r in vals])
            ax.plot(x, y, color=colors[method], marker=markers[method], lw=1.8,
                    ms=4.5, label=display[method])
            ax.fill_between(x, y - e, y + e, color=colors[method], alpha=0.12, linewidth=0)
        title = dataset.replace("-6Context", "").replace("-RegionPaths", "").replace("-OccupationPaths", "")
        ax.set_title(f"({chr(97 + col)}) {title}", loc="left", weight="bold")
        ax.set_xlabel("目标层训练排序数")
        ax.set_ylabel("目标层测试 NLL")
        ax.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "long_tail_target_layers.pdf", bbox_inches="tight")
    fig.savefig(out / "long_tail_target_layers.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_support_sweep(summary_rows, out: Path):
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.sans-serif"] = [FontProperties(fname=str(font_path)).get_name()]
        plt.rcParams["axes.unicode_minus"] = False
    datasets = ["Breakfast-6Context", "SUSHI-RegionPaths", "MovieLens-OccupationPaths"]
    methods = ["Pooled-BT", "Layerwise-BT", "MPC-Rank"]
    display = {"Pooled-BT": "汇聚 BT", "Layerwise-BT": "逐层 BT", "MPC-Rank": "MPC-Rank"}
    colors = {"Pooled-BT": "#8d99a8", "Layerwise-BT": "#d08b45", "MPC-Rank": "#174f83"}
    markers = {"Pooled-BT": "s", "Layerwise-BT": "^", "MPC-Rank": "o"}
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.3), sharex="col")
    for col, dataset in enumerate(datasets):
        subset = [r for r in summary_rows if r["dataset"] == dataset]
        for row_idx, (metric, ylabel) in enumerate((("test_nll", "测试 NLL"),
                                                    ("pair_accuracy", "对象对准确率"))):
            ax = axes[row_idx, col]
            for method in methods:
                vals = sorted((r for r in subset if r["method"] == method),
                              key=lambda r: int(r["records_per_layer"]))
                x = np.array([int(r["records_per_layer"]) for r in vals])
                y = np.array([float(r[f"{metric}_mean"]) for r in vals])
                e = np.array([float(r[f"{metric}_ci95"]) for r in vals])
                ax.plot(x, y, color=colors[method], marker=markers[method], lw=1.8,
                        ms=4.5, label=display[method])
                ax.fill_between(x, y - e, y + e, color=colors[method], alpha=0.12, linewidth=0)
            ax.grid(alpha=0.22)
            ax.set_ylabel(ylabel)
            if row_idx == 0:
                title = dataset.replace("-6Context", "").replace("-RegionPaths", "").replace("-OccupationPaths", "")
                ax.set_title(f"({chr(97 + col)}) {title}", loc="left", weight="bold")
            else:
                ax.set_xlabel("每层训练排序数")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "real_support_sweep.pdf", bbox_inches="tight")
    fig.savefig(out / "real_support_sweep.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def breakfast_dataset() -> Dataset:
    lines = BREAKFAST.read_text(encoding="utf-8-sig").splitlines()
    reader = csv.reader(lines, delimiter=";")
    rows = list(reader)
    layer_names = rows[1]
    records = []
    for uid, row in enumerate(rows[2:]):
        for k, cell in enumerate(row):
            order = np.array([int(x) - 1 for x in cell.split(",")], dtype=int)
            values = np.arange(len(order), 0, -1, dtype=float)
            records.append(Record(uid, k, order, values))
    K = len(layer_names)
    return Dataset("Breakfast-6Context", records, 15, layer_names, np.ones((K, K)), True, 0.65, None)


def sushi_dataset() -> Dataset:
    udata = np.loadtxt(str(SUSHI_DIR / "sushi3.udata"), delimiter="\t", dtype=int)
    raw = (SUSHI_DIR / "sushi3a.5000.10.order").read_text(encoding="utf-8").splitlines()[1:]
    # Current residential region (column 9 in the documented one-based schema).
    regions = udata[:, 8]
    kept = sorted(x for x in np.unique(regions) if np.sum(regions == x) >= 80)
    remap = {x: i for i, x in enumerate(kept)}
    records = []
    for uid, line in enumerate(raw):
        region = int(regions[uid])
        if region not in remap:
            continue
        vals = [int(x) for x in line.split()]
        order = np.array(vals[2:], dtype=int)
        values = np.arange(len(order), 0, -1, dtype=float)
        records.append(Record(uid, remap[region], order, values))
    coords = np.asarray(kept, dtype=float)
    sim = np.exp(-np.abs(coords[:, None] - coords[None, :]) / 2.0)
    return Dataset("SUSHI-RegionPaths", records, 10, [f"region-{x}" for x in kept], sim, False, 0.25, 60)


def movielens_dataset(top_n: int = 35) -> Dataset:
    ratings = np.loadtxt(str(MOVIELENS_DIR / "u.data"), dtype=int)
    user_rows = {}
    occupations = []
    with (MOVIELENS_DIR / "u.user").open("r", encoding="latin-1") as fh:
        for line in fh:
            uid, age, gender, occupation, zipcode = line.rstrip().split("|")
            user_rows[int(uid)] = occupation
            occupations.append(occupation)
    layer_names = sorted(set(occupations))
    layer_map = {x: i for i, x in enumerate(layer_names)}
    item_counts = defaultdict(int)
    for _uid, item, _rating, _ts in ratings:
        item_counts[int(item)] += 1
    top_items = [x for x, _ in sorted(item_counts.items(), key=lambda z: (-z[1], z[0]))[:top_n]]
    item_map = {x: i for i, x in enumerate(top_items)}
    per_user = defaultdict(list)
    for uid, item, rating, _ts in ratings:
        if int(item) in item_map:
            per_user[int(uid)].append((item_map[int(item)], int(rating)))
    records = []
    for uid, vals in per_user.items():
        if len(vals) < 6:
            continue
        vals.sort(key=lambda x: (-x[1], x[0]))
        items = np.array([x[0] for x in vals], dtype=int)
        scores = np.array([x[1] for x in vals], dtype=float)
        records.append(Record(uid, layer_map[user_rows[uid]], items, scores))
    # Label-free layer similarity: cosine similarity of observed pair coverage.
    K = len(layer_names)
    p = top_n * (top_n - 1) // 2
    coverage = np.zeros((K, p), dtype=float)
    def pidx(i, j):
        if i > j:
            i, j = j, i
        return i * (2 * top_n - i - 1) // 2 + (j - i - 1)
    for rec in records:
        for a in range(len(rec.items)):
            for z in range(a + 1, len(rec.items)):
                if rec.values[a] != rec.values[z]:
                    coverage[rec.layer, pidx(int(rec.items[a]), int(rec.items[z]))] = 1.0
    norm = np.linalg.norm(coverage, axis=1)
    sim = coverage @ coverage.T / (norm[:, None] * norm[None, :] + 1e-12)
    return Dataset("MovieLens-OccupationPaths", records, top_n, layer_names, sim, False, 0.25, 25)


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["method"])].append(r)
    out = []
    for (dataset, method), vals in sorted(groups.items()):
        entry = {"dataset": dataset, "method": method, "n_runs": len(vals)}
        for metric in ("kendall_tau", "pair_accuracy", "test_nll"):
            x = np.array([v[metric] for v in vals], dtype=float)
            entry[f"{metric}_mean"] = float(x.mean())
            entry[f"{metric}_std"] = float(x.std(ddof=1)) if len(x) > 1 else 0.0
            entry[f"{metric}_ci95"] = float(1.96 * entry[f"{metric}_std"] / math.sqrt(len(x)))
        out.append(entry)
    return out


def significance(rows):
    out = []
    datasets = sorted({r["dataset"] for r in rows})
    for ds in datasets:
        methods = sorted({r["method"] for r in rows if r["dataset"] == ds and r["method"] != "MPC-Rank"})
        for metric in ("kendall_tau", "pair_accuracy", "test_nll"):
            prop = {r["seed"]: r[metric] for r in rows if r["dataset"] == ds and r["method"] == "MPC-Rank"}
            for method in methods:
                base = {r["seed"]: r[metric] for r in rows if r["dataset"] == ds and r["method"] == method}
                common = sorted(set(prop) & set(base))
                # Positive always denotes an improvement of MPC-Rank.
                diff = np.array([(base[s] - prop[s]) if metric == "test_nll" else (prop[s] - base[s])
                                 for s in common], dtype=float)
                try:
                    p = float(wilcoxon(diff, alternative="greater").pvalue)
                except ValueError:
                    p = 1.0
                out.append({"dataset": ds, "metric": metric, "baseline": method, "n": len(common),
                            "mean_improvement": float(diff.mean()), "wilcoxon_one_sided_p": p})
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_real(summary_rows, out: Path):
    datasets = sorted({r["dataset"] for r in summary_rows})
    methods = ["Borda", "RankCentrality", "Pooled-BT", "MPC-Rank"]
    colors = ["#9aa7b8", "#6f8fab", "#4b7197", "#174f83"]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    metrics = [("kendall_tau", "Kendall $\\tau$"), ("pair_accuracy", "Pairwise accuracy"), ("test_nll", "Test NLL")]
    for ax, (metric, label) in zip(axes, metrics):
        x = np.arange(len(datasets))
        width = 0.18
        for mi, method in enumerate(methods):
            vals, errs = [], []
            for ds in datasets:
                r = next(z for z in summary_rows if z["dataset"] == ds and z["method"] == method)
                vals.append(r[f"{metric}_mean"])
                errs.append(r[f"{metric}_ci95"])
            ax.bar(x + (mi - 1.5) * width, vals, width, yerr=errs, color=colors[mi],
                   edgecolor="white", linewidth=0.5, capsize=2, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels([d.replace("-", "\n", 1) for d in datasets], fontsize=8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "real_data_comparison.pdf", bbox_inches="tight")
    fig.savefig(out / "real_data_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_workflow(out: Path):
    fig, ax = plt.subplots(figsize=(12, 2.65))
    ax.axis("off")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    zh = FontProperties(fname=str(font_path)) if font_path.exists() else None
    boxes = [
        (0.02, "异构网络"),
        (0.22, "候选\n元路径"),
        (0.42, "稀疏元路径\n网络层"),
        (0.62, "跨层耦合\n排名聚合"),
        (0.82, "总体排序与\n主导路径"),
    ]
    for idx, (x, label) in enumerate(boxes):
        fc = "#eaf1f8" if idx != 3 else "#dcecdf"
        ax.add_patch(plt.Rectangle((x, 0.36), 0.15, 0.34, facecolor=fc, edgecolor="#245681", lw=1.3))
        ax.text(x + 0.075, 0.53, label, ha="center", va="center", fontsize=10,
                fontproperties=zh)
        if idx < len(boxes) - 1:
            ax.annotate("", xy=(x + 0.195, 0.53), xytext=(x + 0.155, 0.53),
                        arrowprops=dict(arrowstyle="->", color="#6f8597", lw=1.4))
    ax.text(0.695, 0.20, r"$\sum_{k,\ell}S_{k\ell}\|b^{(k)}-b^{(\ell)}\|_2^2$",
            ha="center", fontsize=11, color="#245681")
    ax.text(0.5, 0.90, "单一机制：相近元路径层应具有相近的相对偏移",
            ha="center", fontsize=11, weight="bold", fontproperties=zh)
    fig.tight_layout()
    fig.savefig(out / "concise_workflow.pdf", bbox_inches="tight")
    fig.savefig(out / "concise_workflow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/metapath_coupling_results")
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(3 if args.smoke else args.seeds))
    steps = 220 if args.smoke else args.steps
    start = time.time()
    datasets = [breakfast_dataset(), sushi_dataset(), movielens_dataset()]
    rows = []
    for ds in datasets:
        print(f"Running {ds.name}: items={ds.n_items}, layers={len(ds.layer_names)}, records={len(ds.records)}", flush=True)
        rows.extend(evaluate_dataset(ds, seeds, steps))
    summary = summarize(rows)
    tests = significance(rows)
    write_csv(args.out / "real_per_seed.csv", rows)
    write_csv(args.out / "real_summary.csv", summary)
    write_csv(args.out / "paired_significance.csv", tests)
    support_caps = {
        "Breakfast-6Context": [3, 5, 8, 12],
        "SUSHI-RegionPaths": [5, 10, 20, 40, 60],
        "MovieLens-OccupationPaths": [3, 5, 10, 15, 25],
    }
    support_rows = []
    for ds in datasets:
        print(f"Support sweep {ds.name}: caps={support_caps[ds.name]}", flush=True)
        support_rows.extend(evaluate_support_sweep(ds, seeds, support_caps[ds.name],
                                                   max(420, steps - 150)))
    support_summary = summarize_support(support_rows)
    support_tests = support_significance(support_rows)
    write_csv(args.out / "support_sweep_per_seed.csv", support_rows)
    write_csv(args.out / "support_sweep_summary.csv", support_summary)
    write_csv(args.out / "support_sweep_significance.csv", support_tests)
    plot_support_sweep(support_summary, args.out)
    long_tail_supports = [0, 1, 3, 5]
    long_tail_rows = []
    for ds in datasets:
        print(f"Long-tail targets {ds.name}: supports={long_tail_supports}", flush=True)
        long_tail_rows.extend(evaluate_long_tail_layers(ds, seeds, long_tail_supports,
                                                        max(420, steps - 150)))
    long_tail_summary = summarize_long_tail(long_tail_rows)
    long_tail_tests = long_tail_significance(long_tail_rows)
    write_csv(args.out / "long_tail_per_seed.csv", long_tail_rows)
    write_csv(args.out / "long_tail_summary.csv", long_tail_summary)
    write_csv(args.out / "long_tail_significance.csv", long_tail_tests)
    plot_long_tail(long_tail_summary, args.out)
    plot_real(summary, args.out)
    plot_workflow(args.out)
    audit = {
        "model": "shared score + layer deviations + graph-Laplacian cross-layer coupling",
        "excluded_modules": ["source reliability", "malicious-source model", "label-flip likelihood",
                             "cross-fitted calibration", "calibration freezing", "nuclear norm"],
        "datasets": [{"name": d.name, "items": d.n_items, "layers": len(d.layer_names),
                      "records": len(d.records), "layer_names": d.layer_names} for d in datasets],
        "seeds": seeds,
        "steps": steps,
        "support_caps": support_caps,
        "support_sweep_methods": ["Pooled-BT", "Layerwise-BT", "MPC-Rank"],
        "long_tail_target_supports": long_tail_supports,
        "long_tail_target_fraction": 0.25,
        "elapsed_seconds": time.time() - start,
        "raw_data_redistribution": "prohibited for MovieLens and SUSHI; retain official download links only",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
