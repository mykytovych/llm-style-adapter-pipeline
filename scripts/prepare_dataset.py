import pandas as pd
import json
import argparse
from pathlib import Path

def format_to_chatml(system_prompt: str, user_prompt: str, response: str) -> dict:
    """
    Converts raw text into the standard ChatML format required by Llama 3 / Unsloth.
    """
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response}
        ]
    }

def process_dataset(input_csv: str, output_jsonl: str, system_prompt: str):
    """
    Reads a CSV file, formats each row, and saves it as a JSONL file.
    """
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Ensure required columns exist
    if not {'prompt', 'response'}.issubset(df.columns):
        raise ValueError("CSV must contain 'prompt' and 'response' columns.")
    
    formatted_data = []
    for _, row in df.iterrows():
        chat_dict = format_to_chatml(
            system_prompt=system_prompt,
            user_prompt=str(row['prompt']),
            response=str(row['response'])
        )
        formatted_data.append(chat_dict)
    
    # Save to JSONL
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Successfully saved {len(formatted_data)} samples to {output_jsonl}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV to Hugging Face JSONL Chat Format")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--system-prompt", type=str, required=True, help="System prompt to define the persona/style")
    
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.system_prompt)