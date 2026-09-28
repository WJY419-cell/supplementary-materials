"""Mechanism-focused synthetic experiments for MPC-Rank.

The design varies target-layer support, relation heterogeneity, and layer-graph
misspecification.  All methods share the same generated comparisons for each
seed, and MPC-Rank selects its coupling strength on an inner validation set.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "code/metapath_coupling_experiment.py"
RESULTS = ROOT / "results"


def load_model():
    spec = importlib.util.spec_from_file_location("mpc_experiment", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pair_index(i: np.ndarray, j: np.ndarray, n_items: int) -> np.ndarray:
    return i * (2 * n_items - i - 1) // 2 + (j - i - 1)


def sample_counts(rng, theta, budgets):
    """Sample and aggregate Bradley--Terry comparisons by layer."""
    n_items, n_layers = theta.shape
    n_pairs = n_items * (n_items - 1) // 2
    wins = np.zeros((n_layers, n_pairs), dtype=float)
    losses = np.zeros_like(wins)
    for k, budget in enumerate(budgets):
        budget = int(budget)
        if budget <= 0:
            continue
        i = rng.randint(0, n_items, size=budget)
        j = rng.randint(0, n_items - 1, size=budget)
        j += j >= i
        lo = np.minimum(i, j)
        hi = np.maximum(i, j)
        idx = pair_index(lo, hi, n_items)
        prob = 1.0 / (1.0 + np.exp(-np.clip(theta[lo, k] - theta[hi, k], -30, 30)))
        outcome = rng.binomial(1, prob)
        np.add.at(wins[k], idx, outcome)
        np.add.at(losses[k], idx, 1 - outcome)
    kk, pp = np.nonzero((wins + losses) > 0)
    # Invert the compact upper-triangle index.
    ii = np.empty(len(pp), dtype=np.int32)
    jj = np.empty(len(pp), dtype=np.int32)
    starts = np.array([x * (2 * n_items - x - 1) // 2 for x in range(n_items)], dtype=int)
    for z, p in enumerate(pp):
        row = int(np.searchsorted(starts, p, side="right") - 1)
        ii[z] = row
        jj[z] = row + 1 + int(p - starts[row])
    return ii, jj, kk.astype(np.int32), wins[kk, pp], losses[kk, pp]


def merge_counts(*parts):
    table = defaultdict(lambda: [0.0, 0.0])
    for i, j, k, win, lose in parts:
        for a, b, c, w, l in zip(i, j, k, win, lose):
            cell = table[(int(a), int(b), int(c))]
            cell[0] += float(w)
            cell[1] += float(l)
    keys = sorted(table)
    return (
        np.array([x[0] for x in keys], dtype=np.int32),
        np.array([x[1] for x in keys], dtype=np.int32),
        np.array([x[2] for x in keys], dtype=np.int32),
        np.array([table[x][0] for x in keys], dtype=float),
        np.array([table[x][1] for x in keys], dtype=float),
    )


def keep_layers(counts, layers):
    mask = np.isin(counts[2], np.asarray(sorted(layers), dtype=np.int32))
    return tuple(x[mask] for x in counts)


def generate_truth(seed, n_items, n_layers, heterogeneity):
    rng = np.random.RandomState(seed + 17011)
    u = rng.normal(size=n_items)
    u = (u - u.mean()) / u.std()
    coords = np.arange(n_layers)
    graph = np.exp(-np.abs(coords[:, None] - coords[None, :]) / 1.5)
    np.fill_diagonal(graph, 0.0)
    lap = np.diag(graph.sum(axis=1)) - graph
    eigval, eigvec = np.linalg.eigh(lap)
    # Draw offsets from the three lowest nonconstant graph-frequency modes.
    modes = eigvec[:, 1:4]
    weights = 1.0 / np.sqrt(np.maximum(eigval[1:4], 1e-8))
    base = (rng.normal(size=(n_items, 3)) * weights) @ modes.T
    base -= base.mean(axis=0, keepdims=True)
    base -= base.mean(axis=1, keepdims=True)
    scale = base.std() if base.std() > 0 else 1.0
    b = float(heterogeneity) * base / scale
    # Match the model's identifiability constraints without reloading the module.
    u -= u.mean()
    b -= b.mean(axis=0, keepdims=True)
    common = b.mean(axis=1, keepdims=True)
    b -= common
    u += common[:, 0]
    u -= u.mean()
    return u, b, u[:, None] + b


def layer_graph(n_layers, corruption, seed):
    coords = np.arange(n_layers)
    true_graph = np.exp(-np.abs(coords[:, None] - coords[None, :]) / 1.5)
    rng = np.random.RandomState(seed + 29021)
    perm = rng.permutation(n_layers)
    wrong_graph = true_graph[np.ix_(perm, perm)]
    observed = (1.0 - corruption) * true_graph + corruption * wrong_graph
    return true_graph, observed


def theta_rmse(score, offsets, truth_theta, target_layers):
    if offsets is None:
        pred = np.repeat(score[:, None], truth_theta.shape[1], axis=1)
    else:
        pred = score[:, None] + offsets
    pred = pred[:, target_layers]
    truth = truth_theta[:, target_layers]
    pred = pred - pred.mean(axis=0, keepdims=True)
    truth = truth - truth.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def safe_tau(x, y):
    value = kendalltau(x, y).correlation
    return float(0.0 if np.isnan(value) else value)


def evaluate_scenario(model, seed, heterogeneity, corruption, target_support,
                      steps, tune_steps, donor_support=500, test_support=1200):
    n_items, n_layers = 30, 8
    target_layers = [2, 5]
    target_set = set(target_layers)
    u_true, _b_true, theta_true = generate_truth(seed, n_items, n_layers, heterogeneity)
    _true_graph, observed_graph = layer_graph(n_layers, corruption, seed)

    total_budgets = np.array([
        target_support if k in target_set else donor_support for k in range(n_layers)
    ], dtype=int)
    fit_budgets = np.floor(0.8 * total_budgets).astype(int)
    val_budgets = total_budgets - fit_budgets
    test_budgets = np.array([test_support if k in target_set else 0 for k in range(n_layers)])

    rng_fit = np.random.RandomState(seed * 100003 + int(heterogeneity * 1000) * 101 + 7)
    rng_val = np.random.RandomState(seed * 100003 + int(heterogeneity * 1000) * 101 + 17)
    rng_test = np.random.RandomState(seed * 100003 + int(heterogeneity * 1000) * 101 + 29)
    inner_train = sample_counts(rng_fit, theta_true, fit_budgets)
    inner_val = sample_counts(rng_val, theta_true, val_budgets)
    full_train = merge_counts(inner_train, inner_val)
    target_test = sample_counts(rng_test, theta_true, test_budgets)

    # Select coupling by a training-internal leave-relation-layer-out task.
    # Two seed-controlled donor layers act as proxy long-tail layers; their
    # inner-training comparisons are withheld and only their validation
    # comparisons determine lambda_s.
    donors = [k for k in range(n_layers) if k not in target_set]
    proxy_rng = np.random.RandomState(seed + 47051)
    proxy_layers = sorted(proxy_rng.choice(donors, size=2, replace=False).tolist())
    tuning_layers = set(range(n_layers)) - set(proxy_layers)
    tuning_train = keep_layers(inner_train, tuning_layers)
    tuning_val = keep_layers(inner_val, proxy_layers)

    lambda_grid = (0.0, 0.0005, 0.002, 0.01, 0.05)
    best_lambda, best_nll = 0.0, float("inf")
    for candidate in lambda_grid:
        cu, cb = model.fit_joint(
            tuning_train, n_items, n_layers, observed_graph,
            lambda_u=0.03, lambda_b=0.005, lambda_s=candidate,
            steps=tune_steps,
        )
        _acc, val_nll = model.pair_metrics(tuning_val, cu, cb)
        if val_nll < best_nll - 1e-12:
            best_lambda, best_nll = candidate, val_nll

    start = time.perf_counter()
    pooled = (model.fit_pooled(full_train, n_items, n_layers, steps=steps), None)
    pooled_time = time.perf_counter() - start
    start = time.perf_counter()
    layerwise = model.fit_layerwise(full_train, n_items, n_layers, steps=max(180, steps - 50))
    layerwise_time = time.perf_counter() - start
    start = time.perf_counter()
    uncoupled = model.fit_joint(
        full_train, n_items, n_layers, observed_graph,
        lambda_u=0.03, lambda_b=0.005, lambda_s=0.0, steps=steps,
    )
    uncoupled_time = time.perf_counter() - start
    start = time.perf_counter()
    mpc = model.fit_joint(
        full_train, n_items, n_layers, observed_graph,
        lambda_u=0.03, lambda_b=0.005, lambda_s=best_lambda, steps=steps,
    )
    mpc_time = time.perf_counter() - start

    rows = []
    for method, (score, offsets), elapsed in (
        ("Pooled-BT", pooled, pooled_time),
        ("Layerwise-BT", layerwise, layerwise_time),
        ("Uncoupled-Joint", uncoupled, uncoupled_time),
        ("MPC-Rank", mpc, mpc_time),
    ):
        acc, nll = model.pair_metrics(target_test, score, offsets)
        rows.append({
            "seed": seed,
            "heterogeneity": heterogeneity,
            "graph_corruption": corruption,
            "target_support": target_support,
            "method": method,
            "target_test_nll": nll,
            "target_pair_accuracy": acc,
            "target_theta_rmse": theta_rmse(score, offsets, theta_true, target_layers),
            "global_tau": "" if method == "Layerwise-BT" else safe_tau(score, u_true),
            "selected_lambda_s": best_lambda if method == "MPC-Rank" else "",
            "proxy_validation_layers": "|".join(str(x) for x in proxy_layers),
            "fit_seconds": elapsed,
        })
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["family"], float(row["heterogeneity"]),
               float(row["graph_corruption"]), int(row["target_support"]), row["method"])
        groups[key].append(row)
    out = []
    for (family, heterogeneity, corruption, support, method), vals in sorted(groups.items()):
        entry = {
            "family": family,
            "heterogeneity": heterogeneity,
            "graph_corruption": corruption,
            "target_support": support,
            "method": method,
            "n_runs": len(vals),
        }
        for metric in ("target_test_nll", "target_pair_accuracy", "target_theta_rmse", "fit_seconds"):
            x = np.array([float(v[metric]) for v in vals])
            entry[f"{metric}_mean"] = float(x.mean())
            entry[f"{metric}_ci95"] = float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))
        if method != "Layerwise-BT":
            x = np.array([float(v["global_tau"]) for v in vals])
            entry["global_tau_mean"] = float(x.mean())
            entry["global_tau_ci95"] = float(1.96 * x.std(ddof=1) / math.sqrt(len(x)))
        else:
            entry["global_tau_mean"] = ""
            entry["global_tau_ci95"] = ""
        out.append(entry)
    return out


def paired_tests(rows):
    out = []
    keys = sorted({(r["family"], float(r["heterogeneity"]),
                    float(r["graph_corruption"]), int(r["target_support"])) for r in rows})
    for family, heterogeneity, corruption, support in keys:
        proposed = {int(r["seed"]): float(r["target_test_nll"]) for r in rows
                    if r["family"] == family and float(r["heterogeneity"]) == heterogeneity
                    and float(r["graph_corruption"]) == corruption
                    and int(r["target_support"]) == support and r["method"] == "MPC-Rank"}
        for baseline in ("Pooled-BT", "Layerwise-BT", "Uncoupled-Joint"):
            base = {int(r["seed"]): float(r["target_test_nll"]) for r in rows
                    if r["family"] == family and float(r["heterogeneity"]) == heterogeneity
                    and float(r["graph_corruption"]) == corruption
                    and int(r["target_support"]) == support and r["method"] == baseline}
            common = sorted(set(proposed) & set(base))
            diff = np.array([base[s] - proposed[s] for s in common])
            try:
                p = float(wilcoxon(diff, alternative="greater").pvalue)
            except ValueError:
                p = 1.0
            out.append({
                "family": family,
                "heterogeneity": heterogeneity,
                "graph_corruption": corruption,
                "target_support": support,
                "baseline": baseline,
                "n": len(common),
                "mean_nll_improvement": float(diff.mean()),
                "wilcoxon_one_sided_p": p,
                "positive_seeds": int(np.sum(diff > 0)),
            })
    return out


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seeds = list(range(3 if args.smoke else args.seeds))
    steps = 100 if args.smoke else args.steps
    tune_steps = 55 if args.smoke else max(130, args.steps // 2)
    model = load_model()
    rows = []
    start = time.time()

    support_levels = [0, 10, 30, 100, 300]
    for support in support_levels:
        print(f"support curve: target comparisons={support}", flush=True)
        for seed in seeds:
            vals = evaluate_scenario(model, seed, 0.6, 0.0, support, steps, tune_steps)
            for row in vals:
                row["family"] = "support"
            rows.extend(vals)

    heterogeneity_levels = [0.0, 0.3, 0.6, 1.0]
    corruption_levels = [0.0, 0.5, 1.0]
    for heterogeneity in heterogeneity_levels:
        for corruption in corruption_levels:
            print(f"boundary: heterogeneity={heterogeneity}, corruption={corruption}", flush=True)
            for seed in seeds:
                vals = evaluate_scenario(model, seed, heterogeneity, corruption, 30,
                                         steps, tune_steps)
                for row in vals:
                    row["family"] = "boundary"
                rows.extend(vals)

    summary = summarize(rows)
    tests = paired_tests(rows)
    suffix = "_smoke" if args.smoke else ""
    write_csv(RESULTS / f"synthetic_per_seed{suffix}.csv", rows)
    write_csv(RESULTS / f"synthetic_summary{suffix}.csv", summary)
    write_csv(RESULTS / f"synthetic_significance{suffix}.csv", tests)
    manifest = {
        "design": "paired mechanism-focused Bradley--Terry simulation",
        "n_items": 30,
        "n_layers": 8,
        "target_layers": [2, 5],
        "donor_training_comparisons_per_layer": 500,
        "target_test_comparisons_per_layer": 1200,
        "support_levels": support_levels,
        "heterogeneity_levels": heterogeneity_levels,
        "graph_corruption_levels": corruption_levels,
        "seeds": seeds,
        "steps": steps,
        "tune_steps": tune_steps,
        "lambda_s_candidates": [0.0, 0.0005, 0.002, 0.01, 0.05],
        "elapsed_seconds": time.time() - start,
    }
    (RESULTS / f"synthetic_manifest{suffix}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
