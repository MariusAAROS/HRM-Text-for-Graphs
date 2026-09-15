"""
Graph Dataset Structural Comparison
====================================

Compares two collections of NetworkX graphs (e.g. two datasets of molecules,
ego-networks, etc.) by computing per-graph structural statistics, comparing
their distributions visually and statistically, and optionally computing
Maximum Mean Discrepancy (MMD) on degree distributions — the standard metric
used in the graph-generation literature (GraphRNN, GRAN, etc.) for comparing
two populations of graphs.

Usage
-----
    from graph_comparison import compare_graph_datasets

    results = compare_graph_datasets(
        graphs_a, graphs_b,
        label_a="Dataset A", label_b="Dataset B",
    )

    results["summary"]   # mean/std/min/max per statistic, per dataset
    results["tests"]     # KS test, Wasserstein distance, Cohen's d per statistic
    results["raw_data"]  # full per-graph stats table (tidy, one row per graph)

Requires: networkx, numpy, pandas, scipy, matplotlib
"""

import warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats


# ----------------------------------------------------------------------
# Per-graph statistics
# ----------------------------------------------------------------------

def _safe_assortativity(G):
    try:
        val = nx.degree_assortativity_coefficient(G)
        return val if val is not None and not np.isnan(val) else np.nan
    except Exception:
        return np.nan


def _largest_component_subgraph(G):
    """Return the subgraph induced by the largest (weakly) connected component."""
    H = G.to_undirected() if G.is_directed() else G
    if H.number_of_nodes() == 0:
        return H
    largest_cc = max(nx.connected_components(H), key=len)
    return H.subgraph(largest_cc).copy()


def _safe_diameter(G):
    try:
        H = _largest_component_subgraph(G)
        if H.number_of_nodes() < 2:
            return np.nan
        return nx.diameter(H)
    except Exception:
        return np.nan


def _safe_avg_shortest_path(G):
    try:
        H = _largest_component_subgraph(G)
        if H.number_of_nodes() < 2:
            return np.nan
        return nx.average_shortest_path_length(H)
    except Exception:
        return np.nan


def _num_components(G):
    return (
        nx.number_weakly_connected_components(G)
        if G.is_directed()
        else nx.number_connected_components(G)
    )


STAT_FUNCS = {
    "num_nodes": lambda G: G.number_of_nodes(),
    "num_edges": lambda G: G.number_of_edges(),
    "density": lambda G: nx.density(G),
    "avg_degree": lambda G: (2 * G.number_of_edges() / G.number_of_nodes())
    if G.number_of_nodes() > 0 else np.nan,
    "avg_clustering": lambda G: nx.average_clustering(G.to_undirected() if G.is_directed() else G),
    "transitivity": lambda G: nx.transitivity(G.to_undirected() if G.is_directed() else G),
    "assortativity": _safe_assortativity,
    "num_triangles": lambda G: sum(
        nx.triangles(G.to_undirected() if G.is_directed() else G).values()
    ) // 3,
    "avg_shortest_path": _safe_avg_shortest_path,
    "diameter": _safe_diameter,
    "num_components": _num_components,
}


def _finite_or_nan(val):
    """Coerce inf/-inf (and non-numeric surprises) to NaN so they don't break
    downstream stats/plots, which expect either real numbers or NaN."""
    try:
        val = float(val)
    except (TypeError, ValueError):
        return np.nan
    return val if np.isfinite(val) else np.nan


def compute_stats(graphs, label):
    """Compute per-graph structural statistics for a list of graphs.

    Degenerate graphs (constant degree, disconnected, tiny, etc.) routinely
    make networkx/numpy emit RuntimeWarnings (divide-by-zero, invalid value)
    for things like assortativity — these are non-fatal and the resulting
    NaN/inf is what we want, so we suppress the noise locally rather than
    let it print or (in stricter environments where warnings are promoted
    to errors) crash the run.
    """
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            for i, G in enumerate(graphs):
                row = {"graph_id": i, "dataset": label}
                for name, func in STAT_FUNCS.items():
                    try:
                        row[name] = _finite_or_nan(func(G))
                    except Exception:
                        row[name] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Distribution comparison (KS test, Wasserstein distance, Cohen's d)
# ----------------------------------------------------------------------

def compare_distributions(df_a, df_b, stat_cols):
    results = []
    for col in stat_cols:
        a_vals = df_a[col].to_numpy(dtype=float)
        a_vals = a_vals[np.isfinite(a_vals)]
        b_vals = df_b[col].to_numpy(dtype=float)
        b_vals = b_vals[np.isfinite(b_vals)]
        if len(a_vals) < 2 or len(b_vals) < 2:
            # Not enough valid (finite) values to compare this statistic —
            # e.g. assortativity is undefined for every graph in a dataset
            # of constant-degree graphs. Skip it rather than crash.
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                ks_stat, ks_p = stats.ks_2samp(a_vals, b_vals)
                wasserstein = stats.wasserstein_distance(a_vals, b_vals)

                n_a, n_b = len(a_vals), len(b_vals)
                pooled_std = np.sqrt(
                    ((n_a - 1) * np.var(a_vals, ddof=1) + (n_b - 1) * np.var(b_vals, ddof=1))
                    / (n_a + n_b - 2)
                ) if (n_a + n_b - 2) > 0 else np.nan
                cohens_d = (
                    (np.mean(a_vals) - np.mean(b_vals)) / pooled_std
                    if pooled_std and np.isfinite(pooled_std) and pooled_std > 0
                    else np.nan
                )

        results.append({
            "statistic": col,
            "mean_a": np.mean(a_vals),
            "mean_b": np.mean(b_vals),
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "wasserstein_distance": wasserstein,
            "cohens_d": cohens_d,
        })
    return pd.DataFrame(results).sort_values("wasserstein_distance", ascending=False)


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def plot_distributions(df, stat_cols, label_a, label_b, save_path=None, bins=20):
    n = len(stat_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(stat_cols):
        ax = axes[i]
        any_plotted = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                for label, color in [(label_a, "tab:blue"), (label_b, "tab:orange")]:
                    vals = df.loc[df["dataset"] == label, col].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) == 0:
                        continue
                    ax.hist(vals, bins=bins, alpha=0.5, label=label, color=color, density=True)
                    any_plotted = True
        ax.set_title(col, fontsize=10)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("probability density", fontsize=8)
        if any_plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, "no finite values", ha="center", va="center",
                     transform=ax.transAxes, fontsize=8, color="gray")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD) on degree distributions
# ----------------------------------------------------------------------

def compute_mmd_degree(graphs_a, graphs_b, sigma=1.0, max_graphs=200, random_state=0):
    """
    MMD^2 between the two datasets' degree distributions, using a Gaussian
    kernel over the Wasserstein (earth mover's) distance between each pair
    of graphs' degree sequences. This mirrors the evaluation protocol used
    in the graph-generation literature (e.g. GraphRNN).

    NOTE: this is O(n^2) in the number of graphs (each pairwise distance is
    itself an EMD computation), so for large datasets it randomly subsamples
    up to `max_graphs` per dataset. Increase max_graphs for a more precise
    (but slower) estimate.
    """
    rng = np.random.default_rng(random_state)

    def subsample(graphs):
        if len(graphs) <= max_graphs:
            return graphs
        idx = rng.choice(len(graphs), size=max_graphs, replace=False)
        return [graphs[i] for i in idx]

    graphs_a = subsample(list(graphs_a))
    graphs_b = subsample(list(graphs_b))

    def degree_seq(G):
        degs = [d for _, d in G.degree()]
        return degs if len(degs) > 0 else [0]

    degs_a = [degree_seq(G) for G in graphs_a]
    degs_b = [degree_seq(G) for G in graphs_b]
    all_degs = degs_a + degs_b
    n_a, n = len(degs_a), len(all_degs)

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = stats.wasserstein_distance(all_degs[i], all_degs[j])
            D[i, j] = D[j, i] = d

    K = np.exp(-(D ** 2) / (2 * sigma ** 2))
    K_aa, K_bb, K_ab = K[:n_a, :n_a], K[n_a:, n_a:], K[:n_a, n_a:]
    mmd2 = K_aa.mean() + K_bb.mean() - 2 * K_ab.mean()
    return float(mmd2)


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def compare_graph_datasets(
    graphs_a,
    graphs_b,
    label_a="Dataset A",
    label_b="Dataset B",
    plot=True,
    save_path=None,
    run_mmd=False,
    mmd_kwargs=None,
):
    """
    Full structural comparison of two lists of NetworkX graphs.

    Parameters
    ----------
    graphs_a, graphs_b : list of nx.Graph / nx.DiGraph
    label_a, label_b   : str, dataset names used in plots/tables
    plot                : bool, whether to render distribution plots
    save_path           : optional path to save the plot figure (e.g. "compare.png")
    run_mmd              : bool, whether to also compute MMD on degree distributions
                            (slower — O(n^2) pairwise EMD; off by default)
    mmd_kwargs           : dict of kwargs passed to compute_mmd_degree

    Returns
    -------
    dict with keys:
        "raw_data" : tidy per-graph stats DataFrame (both datasets)
        "summary"  : mean/std/min/max per statistic per dataset
        "tests"    : KS test, Wasserstein distance, Cohen's d per statistic
        "mmd"      : MMD^2 on degree distributions (only if run_mmd=True)
        "figure"   : matplotlib Figure (only if plot=True)
    """
    df_a = compute_stats(graphs_a, label_a)
    df_b = compute_stats(graphs_b, label_b)
    df = pd.concat([df_a, df_b], ignore_index=True)

    stat_cols = [c for c in df.columns if c not in ("graph_id", "dataset")]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            summary = df.groupby("dataset")[stat_cols].agg(["mean", "std", "min", "max"])
    tests = compare_distributions(df_a, df_b, stat_cols)

    results = {"raw_data": df, "summary": summary, "tests": tests}

    if plot:
        fig = plot_distributions(df, stat_cols, label_a, label_b, save_path=save_path)
        results["figure"] = fig

    if run_mmd:
        mmd_kwargs = mmd_kwargs or {}
        results["mmd"] = compute_mmd_degree(graphs_a, graphs_b, **mmd_kwargs)

    return results


# ----------------------------------------------------------------------
# Demo (only runs if this file is executed directly)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Small synthetic example so you can see the expected output shape.
    rng = np.random.default_rng(0)
    graphs_a = [nx.erdos_renyi_graph(rng.integers(15, 30), 0.15, seed=int(s))
                for s in rng.integers(0, 1e6, size=40)]
    graphs_b = [nx.barabasi_albert_graph(rng.integers(15, 30), 2, seed=int(s))
                for s in rng.integers(0, 1e6, size=40)]

    results = compare_graph_datasets(
        graphs_a, graphs_b,
        label_a="Erdos-Renyi", label_b="Barabasi-Albert",
        plot=True, save_path="demo_comparison.png",
        run_mmd=True,
    )

    print("=== Summary stats ===")
    print(results["summary"])
    print("\n=== Distribution tests (sorted by Wasserstein distance) ===")
    print(results["tests"].to_string(index=False))
    print("\n=== MMD^2 on degree distributions ===")
    print(results["mmd"])
