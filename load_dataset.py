"""
Helper script to load and cache the ELI5 dataset locally.
Run this once to download the dataset, then simple_agent.py will use the cached version.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import pickle

def load_and_cache_dataset():
    """Load ELI5 dataset and cache it locally."""
    cache_file = Path("eli5_dataset_cache.pkl")
    
    if cache_file.exists():
        print(f"📂 Loading dataset from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        print(f"✅ Loaded {len(df)} questions from cache")
        return df
    
    print(f"📥 Downloading ELI5 dataset (first time only)...")
    dataset = load_dataset("sentence-transformers/eli5", split="train")
    
    df = pd.DataFrame({
        'query': dataset['question'],
        'answers': dataset['answer']
    })
    
    print(f"💾 Caching dataset to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"✅ Cached {len(df)} questions")
    return df

if __name__ == "__main__":
    load_and_cache_dataset()
    print("\n🎉 Dataset is ready to use!")
