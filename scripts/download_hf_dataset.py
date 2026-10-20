import argparse
import pandas as pd
from datasets import load_dataset
from pathlib import Path

def main(args):
    """
    Downloads a specific dataset from Hugging Face and saves it as a CSV 
    with 'prompt' and 'response' columns, which are required by our pipeline.
    """
    print(f"Downloading dataset '{args.dataset_name}' from Hugging Face...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset(args.dataset_name, split="train")
    
    # Convert to pandas DataFrame for easier manipulation
    df = dataset.to_pandas()
    
    # Check if specified columns exist and rename them
    if args.prompt_col in df.columns and args.response_col in df.columns:
        df = df.rename(columns={args.prompt_col: "prompt", args.response_col: "response"})
    else:
        raise ValueError(
            f"Columns '{args.prompt_col}' and '{args.response_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
        
    # Keep only the required columns to keep the dataset clean
    df = df[["prompt", "response"]]
    
    # Save to the raw data folder
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Successfully saved {len(df)} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset from Hugging Face and prepare for pipeline")
    parser.add_argument("--dataset_name", type=str, default="harpreetsahota/modern-to-shakesperean-translation", help="Hugging Face dataset name")
    parser.add_argument("--prompt_col", type=str, default="modern", help="Column name to use as user prompt")
    parser.add_argument("--response_col", type=str, default="shakespearean", help="Column name to use as assistant response")
    parser.add_argument("--output_csv", type=str, default="data/raw/shakespeare.csv", help="Path to save the output CSV")
    
    args = parser.parse_args()
    main(args)