# Identifying Emerging AI Technologies in Higher Education

A pipeline for detecting novel, not-yet-mainstream AI tools for students, teachers, and researchers, using GitHub as the data source.

## What it does

- Searches GitHub broadly for AI tools across three user groups
- Filters for topical relevance using an LLM
- Scores each candidate on **novelty** (distance from known mainstream tools) and **growth** (activity/popularity)
- Clusters the top candidates to separate real emerging patterns from noise
- Classifies technique and generates a novelty rationale, with an independent adversarial check

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 1 | `RepoCollection.py` | Collects candidate repos via GitHub REST API |
| 2 | `relevance_filter.py` | LLM relevance check (yes/no/unclear) |
| 3 | `score_novelty.py` | Embeds repos, scores novelty + growth |
| 4 | `cluster_repos.py` | Clusters top candidates, saves a figure |
| 5 | `classify_and_rationale.py` | Technique label + novelty rationale + counter-argument |
| 6 | `final_cleanup.py` | Free rule-based false-positive check |

## Search keywords and topics

**Students** - keywords: `AI tutor`, `adaptive learning`, `intelligent tutoring system`, `knowledge tracing`, `personalized learning LLM`, `automated feedback education`
topics: `intelligent-tutoring-system`, `adaptive-learning`, `edtech`, `education-ai`

**Teachers** - keywords: `AI teaching assistant`, `lesson planning AI`, `curriculum generator`, `classroom analytics`, `AI grading instructor`, `rubric generation LLM`
topics: `teacher-tools`, `edtech`, `classroom-management`

**Researchers** - keywords: `AI literature review`, `research assistant LLM`, `automated systematic review`, `paper summarization research`, `hypothesis generation LLM`, `citation assistant AI`
topics: `research-tools`, `scientific-discovery`, `literature-review`

## Key parameters

- **Novelty threshold:** top 30% (70th percentile) of novelty scores
- **Growth threshold:** top 50% (median) of growth scores - both required together
- **README truncation:** first 1,500 characters per repo - a tool's main purpose is almost always stated in the opening paragraph, so this captures the relevant signal without the noise of install instructions, badges, and license text further down
- **Data source:** GitHub REST API (`/search/repositories` and `/repos/{name}/readme`)

## Setup

```bash
pip install -r requirements.txt
```

Set two environment variables before running:
```bash
export GITHUB_TOKEN=your_token_here
export ANTHROPIC_API_KEY=your_key_here
```

Run the scripts in order (1 to 6). Each stage reads the previous stage's output file and writes its own.

## Tools used

| Tool | Purpose |
|---|---|
| GitHub REST API | Repository search and metadata |
| `requests` | HTTP calls |
| `pandas` | Data handling |
| `sentence-transformers` (MiniLM-L6-v2) | Text embeddings |
| `scikit-learn` (HDBSCAN) | Clustering |
| `umap-learn` | Dimensionality reduction before clustering |
| `matplotlib` | Cluster visualization |
| Claude (Haiku) | Relevance filtering, classification, rationale |

## Limitations

- Closed/commercial tools with no public repo are invisible to this method
- Growth is a single-snapshot proxy, not a true trajectory
- Results are bounded to English-language, well-documented repositories
- Novelty is measured relative to a fixed baseline set, not in absolute terms