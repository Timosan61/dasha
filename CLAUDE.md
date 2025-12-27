# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Dasha** — Instagram Audience Analyzer. Collects followers via Apify, clusters them using OpenAI embeddings + KMeans, and analyzes pain points with GPT-4o-mini.

## Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Data collection
python main.py fetch --target <username> --max-followers 10000
python main.py fetch --skip-profiles  # only followers list

# Analysis pipeline
python main.py analyze                    # full: preprocessing → clustering → GPT
python main.py analyze --skip-clustering  # skip specific steps

# Dashboard
python main.py dashboard
streamlit run frontend/app.py

# Comments analysis
python main.py comments --target <username> --max-reels 20
python main.py analyze-comments

# Export
python main.py export --format both  # CSV + JSON
```

## Architecture

```
Instagram → Apify → [Profile] → Preprocessing (Natasha NLP)
                         ↓
              combined_text → OpenAI Embeddings → KMeans
                         ↓
                    [Cluster] → GPT-4o-mini Analysis
                         ↓
                 [SegmentAnalysis] → Streamlit Dashboard
```

### Services (Microservice pattern)

| Service | File | Purpose |
|---------|------|---------|
| ApifyService | `services/apify_service.py` | Instagram scraping via Apify actors |
| PreprocessingService | `services/preprocessing_service.py` | Russian NLP with Natasha (lemmatization, cleaning) |
| ClusteringService | `services/clustering_service.py` | OpenAI embeddings + sklearn KMeans/HDBSCAN |
| AnalysisService | `services/analysis_service.py` | GPT-4o-mini pain points analysis |

### Database Models (SQLAlchemy)

- **Profile** — Instagram followers with bio, combined_text for clustering
- **Post** — Posts from profiles (caption, hashtags)
- **Cluster** — Topic clusters with keywords
- **SegmentAnalysis** — GPT analysis: segment_name, main_pain, triggers, client_phrase
- **Reel**, **Comment**, **CommentAnalysis** — For comments analysis flow

## Environment Variables

```bash
APIFY_API_TOKEN=<token>           # Required for scraping
APIFY_API_TOKEN_2=<token>         # API key rotation (keys 2-7)
OPENAI_API_KEY=<key>              # Required for embeddings + GPT
INSTAGRAM_TARGET=dasha_samoylina  # Default target
```

## Key Implementation Details

- **API Key Rotation**: `ApifyService.ACTIVE_KEYS` in `services/apify_service.py` — auto-rotates on quota errors
- **Skip existing profiles**: `--skip-existing` flag (default True) checks DB before Apify calls
- **Clustering**: Uses `text-embedding-3-small` (1536 dims), configurable `n_clusters` in main.py
- **Language**: Russian — all NLP uses Natasha, UI in Russian

## File Structure

```
main.py              # CLI entry point (Click)
frontend/app.py      # Streamlit dashboard
database/models.py   # SQLAlchemy models
services/            # 4 microservices
data/dasha.db        # SQLite database
data/raw/            # Raw JSON from Apify
```
