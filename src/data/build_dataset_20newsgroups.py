"""
Building the 20newsgroups dataset and save it as CSV files in data/raw/.

Outputs:
- data/raw/20newsgroups_train.csv
- data/raw/20newsgroups_test.csv

Each CSV contains:
- Text: raw document content
- label: numeric label (0..19)
- label_name: human_readable class name
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd 
from sklearn.datasets import fetch_20newsgroups

#---------------------
# Paths
#---------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

def build_split(subset: str) -> pd.DataFrame:
    """
    Load a dataset split ("train" or "test") and return a DataFrame.
    """
    dataset = fetch_20newsgroups (
            subset=subset,
            remove=(), # keep original text (headers, footers, quotes)
)

    df = pd.DataFrame(
        {
            "text": dataset.data,
            "label": dataset.target,
        }
    )

    # Map numeric labels to readable class names
    label_names = dataset.target_names
    df["label_name"] = df ["label"].map(lambda i: label_names[i])

    #Basic safety checks
    df["text"] = df["text"].astype(str)
    df["label_name"] = df["label_name"].astype(str)

    return df

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    train_df = build_split("train")
    test_df = build_split("test")

    train_path = RAW_DIR / "20newsgroups_train.csv"
    test_path = RAW_DIR / "20newsgroups_test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print("Dataset built successfully")
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Number of classes: {train_df['label_name'].nunique()}")

if __name__ == "__main__":
    main()
    
