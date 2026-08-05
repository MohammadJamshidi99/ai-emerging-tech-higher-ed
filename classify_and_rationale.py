import os
import json
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    raise SystemExit("ANTHROPIC_API_KEY is not set in the environment.")

MODEL = "claude-haiku-4-5-20251001"
IN_FILE = "clustered_repos.csv"
OUT_FILE = "final_shortlist.csv"
HEADERS = {"x-api-key": API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"}

CLASSIFY_PROMPT = """Repo: {full_name}
Description: {description}
Topics: {topics}
README (excerpt): {readme}
Nearest mainstream comparison: {nearest_baseline}

Answer as JSON only, no other text:
{{"technique": "one of: RAG, fine-tuned-model, rule-based-ITS, agentic, other",
  "rationale": "1-2 sentences on what specifically looks different from {nearest_baseline}"}}"""

DEVILS_ADVOCATE_PROMPT = """Repo: {full_name}
Description: {description}
README (excerpt): {readme}

Someone claims this is a novel approach, meaningfully different from established tools
like {nearest_baseline}. Argue the skeptical case: why might this actually just be a
repackaged or minor variation of an existing approach? 1-2 sentences. If you genuinely
can't find a good counter-argument, say so directly instead of forcing one."""


def safe_str(x):
    return "" if pd.isna(x) else str(x)


def call_claude(prompt, max_tokens=200, max_retries=2):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=HEADERS,
                json={"model": MODEL, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]},
                timeout=30,
            )
        except requests.exceptions.RequestException:
            return None

        if r.status_code == 200:
            blocks = r.json().get("content", [])
            text_blocks = [b["text"] for b in blocks if b.get("type") == "text"]
            return " ".join(text_blocks).strip() if text_blocks else None

        if r.status_code == 429 and attempt < max_retries - 1:
            time.sleep(10)
            continue

        print(f"  API error {r.status_code}: {r.text[:200]}")
        return None
    return None


def process_row(row):
    ctx = dict(
        full_name=row["full_name"],
        description=safe_str(row.get("description")) or "(none)",
        topics=safe_str(row.get("topics")) or "(none)",
        readme=safe_str(row.get("readme"))[:1500],
        nearest_baseline=safe_str(row.get("nearest_baseline")) or "known mainstream tools",
    )

    classification_raw = call_claude(CLASSIFY_PROMPT.format(**ctx))
    technique, rationale = "unknown", ""
    if classification_raw:
        try:
            cleaned = classification_raw.strip().strip("`").replace("json\n", "", 1)
            parsed = json.loads(cleaned)
            technique = parsed.get("technique", "unknown")
            rationale = parsed.get("rationale", "")
        except (json.JSONDecodeError, AttributeError):
            rationale = classification_raw[:300]

    counter_argument = call_claude(DEVILS_ADVOCATE_PROMPT.format(**ctx)) or ""
    return technique, rationale, counter_argument


def already_done(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["full_name"])


def main():
    df = pd.read_csv(IN_FILE)
    done = already_done(OUT_FILE)
    remaining = df[~df["full_name"].isin(done)]
    print(f"{len(done)} already processed, {len(remaining)} left")

    header_needed = not os.path.exists(OUT_FILE)
    rows = [row for _, row in remaining.iterrows()]

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(process_row, row): row for row in rows}
        completed = 0
        for fut in as_completed(futures):
            row = futures[fut]
            technique, rationale, counter_argument = fut.result()

            out_row = row.to_dict()
            out_row["technique"] = technique
            out_row["novelty_rationale"] = rationale
            out_row["counter_argument"] = counter_argument
            pd.DataFrame([out_row]).to_csv(OUT_FILE, mode="a", header=header_needed, index=False)
            header_needed = False

            completed += 1
            if completed % 10 == 0:
                print(f"processed {completed}/{len(rows)}...")

    print(f"\ndone - see {OUT_FILE}")


if __name__ == "__main__":
    main()