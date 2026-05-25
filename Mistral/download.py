"""
download_corpus.py

Run this ONCE interactively to download and cache all training corpora to disk.
After this, train_stabilizer.py will load from local disk with no network calls.

Usage:
    python download_corpus.py --save_dir /path/to/your/scratch/corpus
    
On Discovery, use your scratch space, e.g.:
    python download_corpus.py --save_dir /scratch/$USER/stabilizer_corpus
"""

import os
import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Directory to save the corpus shards (use scratch space on HPC).")
    parser.add_argument("--n_c4",   type=int, default=40000, help="Samples from C4")
    parser.add_argument("--n_wiki", type=int, default=40000, help="Samples from Wikipedia")
    parser.add_argument("--n_code", type=int, default=20000, help="Samples from code dataset")
    return parser.parse_args()


def download_and_save(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. C4 — Web text
    # ------------------------------------------------------------------
    c4_path = os.path.join(args.save_dir, "c4")
    if os.path.exists(c4_path):
        print(f"C4 already exists at {c4_path}, skipping.")
    else:
        print(f"Downloading C4 ({args.n_c4} samples)...")
        c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
        samples = []
        for item in c4:
            if len(samples) >= args.n_c4:
                break
            if len(item["text"].strip()) > 50:
                samples.append({"text": item["text"]})
        
        from datasets import Dataset
        Dataset.from_list(samples).save_to_disk(c4_path)
        print(f"C4 saved: {len(samples)} samples -> {c4_path}")

    # ------------------------------------------------------------------
    # 2. Wikipedia
    # ------------------------------------------------------------------
    wiki_path = os.path.join(args.save_dir, "wiki")
    if os.path.exists(wiki_path):
        print(f"Wikipedia already exists at {wiki_path}, skipping.")
    else:
        print(f"Downloading Wikipedia ({args.n_wiki} samples)...")
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        samples = []
        for item in wiki:
            if len(samples) >= args.n_wiki:
                break
            if len(item["text"].strip()) > 50:
                samples.append({"text": item["text"]})

        from datasets import Dataset
        Dataset.from_list(samples).save_to_disk(wiki_path)
        print(f"Wikipedia saved: {len(samples)} samples -> {wiki_path}")

    # ------------------------------------------------------------------
    # 3. Code — try candidates in order until one works
    # ------------------------------------------------------------------
    code_path = os.path.join(args.save_dir, "code")
    if os.path.exists(code_path):
        print(f"Code corpus already exists at {code_path}, skipping.")
    else:
        print(f"Downloading code corpus ({args.n_code} samples)...")
        code_candidates = [
            ("nampdn-ai/tiny-codes",            "prompt",  "train"),
            ("smangrul/code-chat-assistant-v1", "content", "train"),
            ("openai/openai_humaneval",         "prompt",  "test"),
        ]
        samples = []
        for ds_name, text_col, split in code_candidates:
            try:
                ds = load_dataset(ds_name, split=split, streaming=True)
                for item in ds:
                    if len(samples) >= args.n_code:
                        break
                    text = item.get(text_col, "")
                    if len(text.strip()) > 50:
                        samples.append({"text": text})
                if samples:
                    print(f"Code corpus loaded from: {ds_name}")
                    break
            except Exception as e:
                print(f"Could not load {ds_name}: {e}. Trying next...")

        if samples:
            from datasets import Dataset
            Dataset.from_list(samples).save_to_disk(code_path)
            print(f"Code saved: {len(samples)} samples -> {code_path}")
        else:
            print("Warning: No code dataset downloaded. Training will use C4+Wiki only.")

    print("\nAll datasets downloaded. Now run:")
    print(f"  python train_stabilizer.py --corpus_dir {args.save_dir} ...")


if __name__ == "__main__":
    args = parse_args()
    download_and_save(args)
