import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
import umap

IN_FILE = "scored_repos.csv"
OUT_FILE = "clustered_repos.csv"
FIGURE_FILE = "cluster_plot.png"
MODEL_NAME = "all-MiniLM-L6-v2"

NOVELTY_PERCENTILE = 0.70  
GROWTH_PERCENTILE = 0.50   


def combined_text(row):
    parts = [
        str(row.get("description", "") or ""),
        str(row.get("topics", "") or ""),
        str(row.get("readme", "") or "")[:1500],
    ]
    return " ".join(p for p in parts if p and p != "nan")


def plot_clusters(coords_2d, cluster_ids, path):
    fig, ax = plt.subplots(figsize=(9, 7))
    ids = sorted(set(cluster_ids))
    for cid in ids:
        mask = cluster_ids == cid
        label = "noise" if cid == -1 else f"cluster {cid}"
        color = "#bbbbbb" if cid == -1 else None
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=14, alpha=0.7, label=label, color=color)
    ax.set_title("Repository clusters (novel + growing subset)")
    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    if len(ids) <= 12:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(IN_FILE)

    novelty_cutoff = df["novelty_score"].quantile(NOVELTY_PERCENTILE)
    growth_cutoff = df["growth_score"].quantile(GROWTH_PERCENTILE)
    shortlist = df[(df["novelty_score"] >= novelty_cutoff) & (df["growth_score"] >= growth_cutoff)].copy()
    print(f"{len(shortlist)}/{len(df)} repos pass the novelty+growth cutoff")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(shortlist.apply(combined_text, axis=1).tolist(),
                               show_progress_bar=True, normalize_embeddings=True)

    # Clustering directly on raw high-dimensional embeddings tends to collapse
    reduced = umap.UMAP(n_neighbors=15, n_components=10, metric="cosine", random_state=42).fit_transform(embeddings)

    clusterer = HDBSCAN(min_cluster_size=3, min_samples=2, cluster_selection_method="leaf")
    cluster_ids = clusterer.fit_predict(reduced)
    shortlist["cluster_id"] = cluster_ids

    n_clusters = len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)
    n_noise = (cluster_ids == -1).sum()
    print(f"{n_clusters} clusters found, {n_noise} repos flagged as outliers/noise")

    # 2D reduction for plotting 
    # row order, matching cluster_ids) before the dataframe below gets sorted
    coords_2d = umap.UMAP(n_neighbors=15, n_components=2, metric="cosine", random_state=42).fit_transform(embeddings)
    plot_clusters(coords_2d, cluster_ids, FIGURE_FILE)
    print(f"saved cluster figure to {FIGURE_FILE}")

    shortlist = shortlist.sort_values(["cluster_id", "novelty_score"], ascending=[True, False])
    shortlist.to_csv(OUT_FILE, index=False)
    print(f"saved to {OUT_FILE}")


if __name__ == "__main__":
    main()