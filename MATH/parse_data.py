"""
Parse MATH dataset from parquet to JSONL format.

This script converts the competition_math dataset from HuggingFace
into the format required by CS336 Assignment 5.

Usage:
    cd /root/cs336/assignment5-alignment
    python data/MATH/parse_data.py

Input:
    data/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet

Output:
    data/MATH/train.jsonl       - Training data (7500 examples)
    data/MATH/validation.jsonl  - Validation data (5000 examples)
    data/MATH/sft.jsonl         - SFT format data (7500 examples)
"""

import pandas as pd
import json
import re
import os

from cs336_alignment.drgrpo_grader import extract_answer as orig_extract_answer

def extract_answer_robust(passage: str) -> str:
    # 先尝试标准方法
    result = orig_extract_answer(passage)
    if result is not None:
        return result
    
    # 处理 \boxed content（无大括号）的情况
    match = re.search(r'\\boxed\s+([^}\\\s]+)', passage)
    if match:
        return match.group(1).strip()
    
    return None

def parse_math_dataset(parquet_path, output_dir):
    """
    Parse MATH dataset from parquet and save as JSONL files.
    
    Args:
        parquet_path: Path to the parquet file
        output_dir: Directory to save JSONL files
    """
    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample problem:\n{df.iloc[0]['problem'][:200]}...")
    print(f"\nSample solution:\n{df.iloc[0]['solution'][:200]}...")
    
    # Split according to official MATH dataset: 7500 train, 5000 test
    train_df = df.iloc[:7500]
    val_df = df.iloc[7500:]

    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train and validation in standard format
    for split_name, split_df in [('train', train_df), ('validation', val_df)]:
        output_file = os.path.join(output_dir, f'{split_name}.jsonl')
        with open(output_file, 'w') as f:
            for _, row in split_df.iterrows():
                data = {
                    'problem': row['problem'],
                    'solution': row['solution'],
                    'answer': extract_answer_robust(str(row['solution'])),
                    'level': row['level'],
                    'type': row['type']
                }
                f.write(json.dumps(data) + '\n')
        print(f"✓ Saved {split_name}.jsonl: {len(split_df)} examples")

    # Save SFT format
    sft_output_file = os.path.join(output_dir, 'sft.jsonl')
    r1_zero_prompt_template = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: {question}\nAssistant: <think>"
    with open(sft_output_file, 'w') as f:
        for _, row in train_df.iterrows():
            prompt = r1_zero_prompt_template.format(question=row['problem'])
            answer = extract_answer_robust(str(row['solution']))
            output_text = row['solution'] + '</think><answer>' + answer + '</answer>'
            data = {
                'input': prompt,
                'output': output_text
            }
            f.write(json.dumps(data) + '\n')
    print(f"✓ Saved sft.jsonl: {len(train_df)} examples")

if __name__ == '__main__':
    # Default paths (relative to assignment root)
    import sys
    
    # Check if running from assignment root
    if os.path.exists('data/MATH/data'):
        parquet_path = 'data/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet'
        output_dir = 'data/MATH'
    else:
        # Running from data/MATH directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parquet_path = os.path.join(script_dir, 'data/train-00000-of-00001-7320a6f3aba8ebd2.parquet')
        output_dir = script_dir
    
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found at {parquet_path}")
        print("Please download the dataset first:")
        print("  curl -L -o data/MATH/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet \\")
        print("    \"https://huggingface.co/datasets/qwedsacf/competition_math/resolve/main/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet\"")
        sys.exit(1)
    
    parse_math_dataset(parquet_path, output_dir)
    print("\n✓ Done! You can now use train.jsonl, validation.jsonl, and sft.jsonl")
