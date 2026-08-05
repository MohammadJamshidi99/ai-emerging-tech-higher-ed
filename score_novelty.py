import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

IN_FILE = "candidate_repos_filtered.csv"
BASELINE_FILE = "baseline.csv"
OUT_FILE = "scored_repos.csv"
MODEL_NAME = "all-MiniLM-L6-v2"


def combined_text(row):
    """Description first (cleanest signal), then topics, then a truncated
    README - most of a README past the first chunk is install instructions
    and badges, which just add noise to the embedding."""
    parts = [
        str(row.get("description", "") or ""),
        str(row.get("topics", "") or ""),
        str(row.get("readme", "") or "")[:1500],
    ]
    return " ".join(p for p in parts if p and p != "nan")


def main():
    candidates = pd.read_csv(IN_FILE)
    candidates = candidates[candidates["relevant"] != "no"].reset_index(drop=True)
    print(f"{len(candidates)} candidates after dropping non-relevant repos")

    baseline = pd.read_csv(BASELINE_FILE)
    model = SentenceTransformer(MODEL_NAME)

    print("embedding baseline...")
    baseline_emb = model.encode(baseline.apply(combined_text, axis=1).tolist(), normalize_embeddings=True)

    print("embedding candidates...")
    cand_emb = model.encode(
        candidates.apply(combined_text, axis=1).tolist(),
        show_progress_bar=True, normalize_embeddings=True,
    )

    # embeddings normalized
    novelty_scores, nearest_baseline = [], []
    for i, row in candidates.iterrows():
        repo_groups = set(str(row["groups"]).split(", "))
        mask = baseline["group"].isin(repo_groups)
        if not mask.any():
            mask = pd.Series([True] * len(baseline))

        sims = cand_emb[i] @ baseline_emb[mask.values].T
        best_idx = sims.argmax()
        novelty_scores.append(1 - sims[best_idx])
        nearest_baseline.append(baseline[mask.values].iloc[best_idx]["name"])

    candidates["novelty_score"] = novelty_scores
    candidates["nearest_baseline"] = nearest_baseline

    # growth proxy
    candidates["pushed_at"] = pd.to_datetime(candidates["pushed_at"], errors="coerce")
    days_since_push = (pd.Timestamp.now(tz="UTC") - candidates["pushed_at"]).dt.days
    recency_bonus = (days_since_push < 180).astype(float) * 0.3
    candidates["growth_score"] = np.log1p(candidates["stars"].fillna(0)) / 10 + recency_bonus

    candidates = candidates.sort_values(["novelty_score", "growth_score"], ascending=False)
    candidates.to_csv(OUT_FILE, index=False)
    print(f"\nsaved {len(candidates)} scored repos to {OUT_FILE}")
    print(candidates[["full_name", "groups", "novelty_score", "growth_score", "nearest_baseline"]].head(15))


if __name__ == "__main__":
    main()