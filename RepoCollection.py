"""
Collects candidate GitHub repositories for the AI-in-education search.
Runs keyword and topic queries per user group through the GitHub Search API,
then pulls README text and basic activity stats for each hit.

Requires a GitHub personal access token set as GITHUB_TOKEN in the environment.
"""

import os
import json
import time
import base64
import requests
import pandas as pd

TOKEN = os.environ.get("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {TOKEN}", "Accept": "application/vnd.github+json"}
BASE = "https://api.github.com"

SEEN_FILE = "seen_repos.json"
OUT_FILE = "candidate_repos.csv"

# One keyword/topic set per user group. Every match tagged with group

KEYWORDS = {
    "student": [
        "AI tutor", "adaptive learning", "intelligent tutoring system",
        "knowledge tracing", "personalized learning LLM", "automated feedback education",
    ],
    "teacher": [
        "AI teaching assistant", "lesson planning AI", "curriculum generator",
        "classroom analytics", "AI grading instructor", "rubric generation LLM",
    ],
    "researcher": [
        "AI literature review", "research assistant LLM", "automated systematic review",
        "paper summarization research", "hypothesis generation LLM", "citation assistant AI",
    ],
}

TOPICS = {
    "student": ["intelligent-tutoring-system", "adaptive-learning", "edtech", "education-ai"],
    "teacher": ["teacher-tools", "edtech", "classroom-management"],
    "researcher": ["research-tools", "scientific-discovery", "literature-review"],
}

# Cheap keyword filter applied before enrichment, to skip obvious noise

NEGATIVE_KEYWORDS = [
    "chatgpt wrapper", "gpt wrapper", "homework helper", "homework assignment",
    "course project", "cs50", "assignment solution", "leetcode",
]


def is_noise(repo):
    text = f"{repo.get('name', '')} {repo.get('description') or ''}".lower()
    return any(bad in text for bad in NEGATIVE_KEYWORDS)


def get_with_retries(url, params=None, max_retries=3):
    """GET request with retries on dropped connections."""
    for attempt in range(max_retries):
        try:
            return requests.get(url, headers=HEADERS, params=params, timeout=15)
        except requests.exceptions.RequestException:
            wait = 5 * (attempt + 1)
            print(f"  network error, retrying in {wait}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
    print(f"  gave up on {url}")
    return None


def search_repos(query, max_pages=3):
    """Runs one search query, paginated, returns a list of repo dicts."""
    results = []
    for page in range(1, max_pages + 1):
        r = get_with_retries(
            f"{BASE}/search/repositories",
            params={"q": query, "per_page": 100, "page": page},
        )
        if r is None or r.status_code != 200:
            break
        items = r.json().get("items", [])
        if not items:
            break
        results.extend(items)
        time.sleep(2)  
    return results


def load_seen():
    if not os.path.exists(SEEN_FILE):
        return {}, set()
    with open(SEEN_FILE) as f:
        raw = json.load(f)
    seen = {fn: (entry["repo"], set(entry["groups"])) for fn, entry in raw["repos"].items()}
    return seen, set(raw["done_queries"])


def save_seen(seen, done_queries):
    payload = {
        "repos": {fn: {"repo": repo, "groups": sorted(groups)} for fn, (repo, groups) in seen.items()},
        "done_queries": sorted(done_queries),
    }
    with open(SEEN_FILE, "w") as f:
        json.dump(payload, f)


def collect_candidates():
    """Runs every keyword/topic query across all groups and merges the results.
    Checkpoints after each query so an interrupted run can resume."""
    seen, done_queries = load_seen()
    if seen:
        print(f"resuming: {len(seen)} repos, {len(done_queries)} queries already done")

    all_queries = []
    for group, keywords in KEYWORDS.items():
        all_queries += [(group, kw, "keyword") for kw in keywords]
        all_queries += [(group, topic, "topic") for topic in TOPICS[group]]

    dropped = 0
    for group, term, kind in all_queries:
        query_key = f"{group}|{kind}|{term}"
        if query_key in done_queries:
            continue

        print(f"[{group}] {kind}: {term}")
        query = f"topic:{term}" if kind == "topic" else term
        for repo in search_repos(query):
            if is_noise(repo):
                dropped += 1
                continue
            full_name = repo["full_name"]
            if full_name not in seen:
                seen[full_name] = (repo, set())
            seen[full_name][1].add(group)

        done_queries.add(query_key)
        save_seen(seen, done_queries)

    print(f"dropped {dropped} noise hits, {len(seen)} unique repos found")
    return seen


def get_readme(full_name):
    r = get_with_retries(f"{BASE}/repos/{full_name}/readme")
    if r is None or r.status_code != 200:
        return ""
    try:
        return base64.b64decode(r.json().get("content", "")).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def already_enriched(path):
    if not os.path.exists(path):
        return set()
    return set(pd.read_csv(path)["full_name"])


def enrich(seen, path=OUT_FILE):
    """Pulls README text and activity stats for each collected repo,
    writing incrementally so an interrupted run can resume."""
    done = already_enriched(path)
    remaining = [(fn, r, g) for fn, (r, g) in seen.items() if fn not in done]
    print(f"{len(done)} already enriched, {len(remaining)} left")

    header_needed = not os.path.exists(path)
    for i, (full_name, repo, groups) in enumerate(remaining):
        if i % 25 == 0:
            print(f"enriching {i}/{len(remaining)}...")

        row = {
            "full_name": full_name,
            "groups": ", ".join(sorted(groups)),
            "description": repo.get("description") or "",
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "created_at": repo["created_at"],
            "pushed_at": repo["pushed_at"],
            "topics": ", ".join(repo.get("topics", [])),
            "readme": get_readme(full_name),
        }
        pd.DataFrame([row]).to_csv(path, mode="a", header=header_needed, index=False)
        header_needed = False
        time.sleep(0.3)

    return pd.read_csv(path)


if __name__ == "__main__":
    candidates = collect_candidates()
    df = enrich(candidates)
    print(f"\n{len(df)} repos total in {OUT_FILE}")