import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL = "claude-haiku-4-5-20251001"
IN_FILE = "candidate_repos.csv"
OUT_FILE = "candidate_repos_filtered.csv"

PROMPT = """Here is a GitHub repo's description, topics, and README (may be truncated).

Description: {description}
Topics: {topics}
README (excerpt): {readme}

Question: is this an AI-based tool intended for {group}s in an educational /
higher-education context (e.g. a tutoring tool, teaching assistant tool, or
research-support tool - not a generic library, course-assignment, or
unrelated project)?

Answer with exactly one word: yes, no, or unclear."""


def safe_str(x):
    """pandas returns NaN (a float) for empty cells, not None or ''."""
    return "" if pd.isna(x) else str(x)


def classify(description, topics, readme, group):
    text = PROMPT.format(
        description=safe_str(description) or "(none)",
        topics=safe_str(topics) or "(none)",
        readme=safe_str(readme)[:1500],
        group=group,
    )

    for attempt in range(3):
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={"model": MODEL, "max_tokens": 5, "messages": [{"role": "user", "content": text}]},
                timeout=20,
            )
        except requests.exceptions.RequestException:
            time.sleep(5 * (attempt + 1))
            continue

        if r.status_code != 200:
            print(f"  API error {r.status_code}: {r.text[:200]}")
            return "unclear"

        blocks = r.json().get("content", [])
        text_blocks = [b["text"] for b in blocks if b.get("type") == "text"]
        answer = (text_blocks[0].strip().lower() if text_blocks else "")
        return answer if answer in ("yes", "no", "unclear") else "unclear"

    return "unclear"  


def already_done(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["full_name"])


def main():
    df = pd.read_csv(IN_FILE)
    done = already_done(OUT_FILE)
    remaining = df[~df["full_name"].isin(done)]
    print(f"{len(done)} already classified, {len(remaining)} left")

    header_needed = not os.path.exists(OUT_FILE)
    rows = list(remaining.iterrows())
    completed = 0

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = {}
        for _, row in rows:
            primary_group = str(row["groups"]).split(",")[0].strip()
            fut = pool.submit(classify, row["description"], row["topics"], row["readme"], primary_group)
            futures[fut] = row

        for fut in as_completed(futures):
            row = futures[fut]
            out_row = row.to_dict()
            out_row["relevant"] = fut.result()
            pd.DataFrame([out_row]).to_csv(OUT_FILE, mode="a", header=header_needed, index=False)
            header_needed = False

            completed += 1
            if completed % 50 == 0:
                print(f"classifying {completed}/{len(rows)}...")

    result = pd.read_csv(OUT_FILE)
    print(f"\ndone: {(result['relevant'] == 'yes').sum()} yes, "
          f"{(result['relevant'] == 'no').sum()} no, "
          f"{(result['relevant'] == 'unclear').sum()} unclear")


if __name__ == "__main__":
    main()