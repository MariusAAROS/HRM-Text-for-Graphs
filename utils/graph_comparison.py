"""
Graph Dataset Structural Comparison
====================================

Compares two OR MORE collections of NetworkX graphs (e.g. datasets of
molecules, ego-networks, etc.) by computing per-graph structural statistics,
comparing their distributions visually and statistically, and optionally
computing Maximum Mean Discrepancy (MMD) on degree distributions — the
standard metric used in the graph-generation literature (GraphRNN, GRAN,
etc.) for comparing populations of graphs.

Usage (2 or more datasets)
---------------------------
    from graph_comparison import GraphDatasetComparison

    comp = GraphDatasetComparison({
        "Dataset A": graphs_a,
        "Dataset B": graphs_b,
        "Dataset C": graphs_c,          # any number of datasets, 2+
    })
    results = comp.compare(run_mmd=True)

    results["summary"]         # mean/std/min/max per statistic, per dataset
    results["pairwise_tests"]  # KS test, Wasserstein, Cohen's d for every dataset pair
    results["raw_data"]        # tidy per-graph stats table (one row per graph)
    results["mmd_matrix"]      # pairwise MMD^2 matrix (only if run_mmd=True)

Usage (exactly 2 datasets, old API)
-------------------------------------
    from graph_comparison import compare_graph_datasets

    results = compare_graph_datasets(
        graphs_a, graphs_b,
        label_a="Dataset A", label_b="Dataset B",
    )
    results["tests"]  # KS test, Wasserstein distance, Cohen's d per statistic

Requires: networkx, numpy, pandas, scipy, matplotlib
"""

import warnings
import itertools
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

def plot_distributions(df, stat_cols, labels, save_path=None, bins=20, ncols=3,
                        log_y="auto", log_y_threshold=20):
    """
    Parameters
    ----------
    log_y : "auto" | True | False
        "auto" (default): each subplot independently switches to a log
        y-axis if one dataset's histogram peak is more than `log_y_threshold`
        times taller than the smallest nonzero bar in that subplot — this is
        what keeps a low, spread-out distribution from being flattened to
        invisibility next to another dataset's sharp spike.
        True: always use log y-axis. False: never (old behavior).
    log_y_threshold : float
        Peak-to-smallest-bar ratio that triggers auto log-scaling.
    """
    n = len(stat_cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    cmap = plt.get_cmap("tab10" if len(labels) <= 10 else "tab20")
    colors = {label: cmap(i % cmap.N) for i, label in enumerate(labels)}

    for i, col in enumerate(stat_cols):
        ax = axes[i]
        any_plotted = False
        heights = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                # Collect finite values per dataset first, so every dataset
                # in this subplot shares the same bin edges — without this,
                # each dataset's histogram silently uses its own x-range,
                # making the bars not directly comparable.
                per_label_vals = {}
                all_vals = []
                for label in labels:
                    vals = df.loc[df["dataset"] == label, col].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if len(vals) == 0:
                        continue
                    per_label_vals[label] = vals
                    all_vals.append(vals)

                if all_vals:
                    combined = np.concatenate(all_vals)
                    lo, hi = combined.min(), combined.max()
                    if lo == hi:
                        pad = 0.5 if lo == 0 else abs(lo) * 0.05 + 1e-9
                        lo, hi = lo - pad, hi + pad
                    shared_edges = np.linspace(lo, hi, bins + 1)

                    for label in labels:
                        vals = per_label_vals.get(label)
                        if vals is None:
                            continue
                        counts, _, _ = ax.hist(vals, bins=shared_edges, alpha=0.5,
                                                label=label, color=colors[label], density=True)
                        heights.append(np.asarray(counts))
                        any_plotted = True

                if any_plotted and log_y in (True, "auto"):
                    positive = np.concatenate(heights)
                    positive = positive[positive > 0]
                    if len(positive) > 0:
                        ratio = positive.max() / positive.min()
                        if log_y is True or ratio > log_y_threshold:
                            ax.set_yscale("log")

        ax.set_title(col, fontsize=10)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("probability density" + (" (log)" if ax.get_yscale() == "log" else ""), fontsize=8)
        if any_plotted:
            ax.legend(fontsize=7)
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
# Main entry point: GraphDatasetComparison (supports 2..N datasets)
# ----------------------------------------------------------------------

class GraphDatasetComparison:
    """
    Compare structural statistics across two or more collections of
    NetworkX graphs (e.g. molecules, ego-networks, citation graphs, etc.).

    Usage
    -----
        comp = GraphDatasetComparison({
            "Dataset A": graphs_a,
            "Dataset B": graphs_b,
            "Dataset C": graphs_c,          # any number of datasets, 2+
        })

        comp.summary()          # mean/std/min/max per stat, per dataset
        comp.pairwise_tests()   # KS test, Wasserstein, Cohen's d for EVERY pair of datasets
        comp.plot()             # one grid of overlaid histograms, all datasets together
        comp.mmd_matrix()       # optional, pairwise MMD^2 matrix on degree distributions

        # or get everything in one call:
        results = comp.compare(run_mmd=True)
    """

    def __init__(self, datasets):
        """
        Parameters
        ----------
        datasets : dict[str, list[nx.Graph]]
            Mapping from dataset label -> list of NetworkX graphs. Must
            contain at least 2 datasets.
        """
        if len(datasets) < 2:
            raise ValueError("Need at least 2 datasets to compare.")
        self.datasets = dict(datasets)
        self.labels = list(self.datasets.keys())
        self._raw_data = None  # computed lazily, cached

    @property
    def raw_data(self):
        """Tidy per-graph stats DataFrame across all datasets (one row per graph)."""
        if self._raw_data is None:
            dfs = [compute_stats(graphs, label) for label, graphs in self.datasets.items()]
            self._raw_data = pd.concat(dfs, ignore_index=True)
        return self._raw_data

    @property
    def stat_cols(self):
        return [c for c in self.raw_data.columns if c not in ("graph_id", "dataset")]

    def summary(self):
        """Mean/std/min/max per statistic, per dataset."""
        df = self.raw_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                return (
                    df.groupby("dataset")[self.stat_cols]
                    .agg(["mean", "std", "min", "max"])
                    .loc[self.labels]
                )

    def pairwise_tests(self):
        """
        KS test, Wasserstein distance, and Cohen's d per statistic, for
        every pair of datasets (C(n, 2) pairs). Sorted within each pair by
        Wasserstein distance, so the most-divergent statistics surface first.
        """
        df = self.raw_data
        rows = []
        for label_a, label_b in itertools.combinations(self.labels, 2):
            df_a = df[df["dataset"] == label_a]
            df_b = df[df["dataset"] == label_b]
            pair = compare_distributions(df_a, df_b, self.stat_cols)
            pair.insert(0, "dataset_b", label_b)
            pair.insert(0, "dataset_a", label_a)
            rows.append(pair)
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def plot(self, save_path=None, bins=20, ncols=3, log_y="auto", log_y_threshold=20):
        """Grid of overlaid distribution plots, one subplot per statistic, all datasets together.

        log_y : "auto" (default) switches a subplot to a log y-axis when one
        dataset's peak is much taller than another's — otherwise a
        low/spread-out distribution can get visually flattened to nothing
        next to a sharp spike. Pass True/False to force it on/off everywhere.
        """
        return plot_distributions(self.raw_data, self.stat_cols, self.labels,
                                   save_path=save_path, bins=bins, ncols=ncols,
                                   log_y=log_y, log_y_threshold=log_y_threshold)

    def mmd_matrix(self, sigma=1.0, max_graphs=200, random_state=0):
        """
        Symmetric DataFrame of pairwise MMD^2 (degree-distribution based)
        between every pair of datasets. Diagonal is 0 by definition.

        NOTE: O(n^2) per pair in number of graphs (see compute_mmd_degree) —
        for many datasets and/or large collections this can be slow.
        """
        mat = pd.DataFrame(0.0, index=self.labels, columns=self.labels)
        for label_a, label_b in itertools.combinations(self.labels, 2):
            val = compute_mmd_degree(
                self.datasets[label_a], self.datasets[label_b],
                sigma=sigma, max_graphs=max_graphs, random_state=random_state,
            )
            mat.loc[label_a, label_b] = val
            mat.loc[label_b, label_a] = val
        return mat

    def compare(self, plot=True, save_path=None, run_mmd=False, mmd_kwargs=None, plot_kwargs=None):
        """
        Run the full comparison and return a dict with keys:
            "raw_data"        : tidy per-graph stats DataFrame, all datasets
            "summary"         : mean/std/min/max per statistic per dataset
            "pairwise_tests"  : KS test, Wasserstein, Cohen's d for every dataset pair
            "mmd_matrix"      : pairwise MMD^2 matrix (only if run_mmd=True)
            "figure"          : matplotlib Figure (only if plot=True)
        """
        results = {
            "raw_data": self.raw_data,
            "summary": self.summary(),
            "pairwise_tests": self.pairwise_tests(),
        }
        if plot:
            plot_kwargs = plot_kwargs or {}
            results["figure"] = self.plot(save_path=save_path, **plot_kwargs)
        if run_mmd:
            mmd_kwargs = mmd_kwargs or {}
            results["mmd_matrix"] = self.mmd_matrix(**mmd_kwargs)
        return results


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
    Backward-compatible convenience wrapper for comparing exactly two
    datasets. For 3+ datasets, use GraphDatasetComparison directly.

    Returns
    -------
    dict with keys:
        "raw_data" : tidy per-graph stats DataFrame (both datasets)
        "summary"  : mean/std/min/max per statistic per dataset
        "tests"    : KS test, Wasserstein distance, Cohen's d per statistic
        "mmd"      : MMD^2 on degree distributions (only if run_mmd=True)
        "figure"   : matplotlib Figure (only if plot=True)
    """
    comp = GraphDatasetComparison({label_a: graphs_a, label_b: graphs_b})
    results = comp.compare(plot=plot, save_path=save_path, run_mmd=run_mmd, mmd_kwargs=mmd_kwargs)

    tests = results.pop("pairwise_tests")
    results["tests"] = tests.drop(columns=["dataset_a", "dataset_b"], errors="ignore")

    if "mmd_matrix" in results:
        results["mmd"] = float(results.pop("mmd_matrix").loc[label_a, label_b])

    return results


# ----------------------------------------------------------------------
# Demo (only runs if this file is executed directly)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Small synthetic example so you can see the expected output shape.
    rng = np.random.default_rng(0)

    def make_graphs(gen_func, n_graphs=40):
        return [gen_func(int(rng.integers(15, 30)), seed=int(s))
                for s in rng.integers(0, 1_000_000, size=n_graphs)]

    graphs_a = make_graphs(lambda n, seed: nx.erdos_renyi_graph(n, 0.15, seed=seed))
    graphs_b = make_graphs(lambda n, seed: nx.barabasi_albert_graph(n, 2, seed=seed))
    graphs_c = make_graphs(lambda n, seed: nx.watts_strogatz_graph(n, 4, 0.1, seed=seed))

    print("############ N-dataset class API (3 datasets) ############")
    comp = GraphDatasetComparison({
        "Erdos-Renyi": graphs_a,
        "Barabasi-Albert": graphs_b,
        "Watts-Strogatz": graphs_c,
    })
    results = comp.compare(plot=True, save_path="demo_comparison.png", run_mmd=True,
                            mmd_kwargs={"max_graphs": 40})

    print("=== Summary stats ===")
    print(results["summary"])
    print("\n=== Pairwise tests (top rows per pair, sorted by Wasserstein distance) ===")
    print(results["pairwise_tests"].groupby(["dataset_a", "dataset_b"]).head(3).to_string(index=False))
    print("\n=== MMD^2 matrix on degree distributions ===")
    print(results["mmd_matrix"])

    print("\n############ Backward-compatible 2-dataset function API ############")
    old_results = compare_graph_datasets(
        graphs_a, graphs_b,
        label_a="Erdos-Renyi", label_b="Barabasi-Albert",
        plot=False, run_mmd=True,
    )
    print("=== tests (top 3 by Wasserstein distance) ===")
    print(old_results["tests"].head(3).to_string(index=False))
    print("MMD:", old_results["mmd"])
