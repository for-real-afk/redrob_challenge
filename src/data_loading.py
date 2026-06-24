"""
Data loading utilities for candidate JSONL files (supporting both raw and gzipped formats).
"""
import gzip
import json
import os

def load_candidates(file_path):
    """
    Loads candidates from a .jsonl or .jsonl.gz file.
    Yields candidate dictionaries one by one to keep memory consumption low.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Candidate file not found at: {file_path}")
        
    is_gz = file_path.endswith(".gz")
    
    open_func = gzip.open if is_gz else open
    mode = "rt" if is_gz else "r"
    
    try:
        with open_func(file_path, mode, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield json.loads(line)
    except Exception as e:
        raise IOError(f"Error reading candidate data file: {e}")
