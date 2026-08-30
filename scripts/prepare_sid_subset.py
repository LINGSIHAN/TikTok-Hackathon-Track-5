import argparse
import hashlib
from pathlib import Path
import random
import sys
import pandas as pd
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare SID_Set dataset subset and manifest.")
    parser.add_argument("--total", type=int, default=10000, help="Total dataset size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def process_and_build_manifest(total: int = 10000, seed: int = 42):
    random.seed(seed)
    
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    raw_authentic_dir = data_dir / "raw" / "authentic"
    raw_generated_dir = data_dir / "raw" / "generated"
    processed_dir = data_dir / "processed"
    
    raw_authentic_dir.mkdir(parents=True, exist_ok=True)
    raw_generated_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Collect files from local storage if available
    auth_files = sorted(list(raw_authentic_dir.glob("*.[jJ][pP][gG]")) + list(raw_authentic_dir.glob("*.[pP][nN][gG]")))
    gen_files = sorted(list(raw_generated_dir.glob("*.[jJ][pP][gG]")) + list(raw_generated_dir.glob("*.[pP][nN][gG]")))

    records = []
    seen_hashes = set()

    for file_path, label in [(f, 0) for f in auth_files] + [(f, 1) for f in gen_files]:
        rel_path = file_path.relative_to(repo_root).as_posix()
        
        with open(file_path, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)

        try:
            with Image.open(file_path) as img:
                w, h = img.size
        except Exception:
            continue

        records.append({
            "path": rel_path,
            "label": label,
            "dataset": "SID_Set",
            "source_id": file_path.stem,
            "width": w,
            "height": h,
            "sha256": file_hash
        })

    if not records:
        print("No raw images found under data/raw/. Initializing empty template manifest.")
        df = pd.DataFrame(columns=["path", "label", "split", "dataset", "source_id", "width", "height", "sha256"])
        df.to_csv(processed_dir / "manifest.csv", index=False)
        return

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split proportions: 80% train, 10% val, 10% test
    n_total = len(df)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    splits = []
    for idx in range(n_total):
        if idx < n_train:
            splits.append("train")
        elif idx < n_train + n_val:
            splits.append("val")
        else:
            splits.append("test")

    df["split"] = splits
    manifest_path = processed_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"Manifest created successfully at {manifest_path} with {len(df)} records.")


if __name__ == "__main__":
    args = parse_args()
    process_and_build_manifest(total=args.total, seed=args.seed)