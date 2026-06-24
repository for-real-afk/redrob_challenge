# Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge

An explainable and deterministic ranking system designed to discover the top 100 candidates matching the Senior AI Engineer role at Redrob, respecting all resource and accuracy constraints.

## Project Structure

- `rank.py`: entrypoint for ranking candidates
- `src/`: source code for features, scoring, honeypot detection, and disqualifiers
- `precompute/`: scripts for offline feature/embeddings computation
- `tests/`: unit tests for evaluation and compliance validation
- `app/`: Streamlit sandbox application
