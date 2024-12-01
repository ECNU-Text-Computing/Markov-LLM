import torch
import re
import pandas as pd
import numpy as np
import argparse
import time
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompt import patent_importance
from data_loader import DBLPDataset
from tqdm import tqdm
import ast
def clean_ref_abstracts(data):
    """Converts string representation of a list to a list and replaces 'nan' with None."""
    if isinstance(data, str):
        try:
            data = ast.literal_eval(data)
        except (ValueError, SyntaxError):
            return None
    if isinstance(data, list):
        return [None if x == 'nan' else x for x in data]

def safe_eval(expr):
    """Safely evaluates Python expressions from strings handling errors gracefully."""
    if isinstance(expr, list):
        return [eval(x) if x is not None else None for x in expr]
    try:
        return eval(expr) if expr is not None else None
    except Exception as e:
        print(f"Error evaluating {expr}: {e}")
        return None

if __name__ == '__main__':
    # Setup command-line arguments
    parser = argparse.ArgumentParser(description="Run GLM model for generating differences.")
    parser.add_argument('--phase', type=str, required=True, help="Phase to control which model to use")
    parser.add_argument('--data', type=str, required=True, help="Phase to control which data to use")
    args = parser.parse_args()

    # Determine the computing device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer based on the phase specified
    if args.phase in ['chatglm', 'chatglm3-6b-32k', 'chatglm-base']:
        tokenizer = AutoTokenizer.from_pretrained(f"{args.phase}", trust_remote_code=True)
    else:
        raise ValueError("Unsupported phase argument")

    # Read and process the dataset
    df = pd.read_csv(f'data/{args.data}_last.csv', sep='\t', header=None,
                       names=['paper_id', 'ni', 'nj', 'nk', 'd', 'd_new',
            'citation', 'title', 'abstract', 'year',
            'author_count', 'ref_titles', 'ref_ids', 'ref_abs','num'])

    df['ref_abstract'] = df['ref_abs'].apply(clean_ref_abstracts)
    df = df.explode('ref_abstract').reset_index(drop=True)
    df.dropna(subset=['abstract', 'ref_abstract'], inplace=True)
    df['input'] = df.apply(lambda row: patent_importance(row['abstract'], row['ref_abstract']), axis=1)
    df = df[['abstract', 'ref_abstract', 'd', 'input']].reset_index(drop=True)
    df['input_length'] = df['input'].apply(len)
    df = df.sort_values(by='input_length').reset_index(drop=True)

    # Prepare data loader
    dataset = DBLPDataset(df, tokenizer)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    tqdm(data_loader, desc='Processing')

    # Load model and set to evaluation mode
    model = AutoModelForCausalLM.from_pretrained(f"{args.phase}", trust_remote_code=True).to(
        device)
    model.eval()
    gen_kwargs = {"max_length": 8192, "num_beams": 1, "do_sample": True, "top_p": 0.8, "temperature": 0.8}

    # Generating and processing outputs
    results_df = pd.DataFrame()
    global_index = 0
    for batch in data_loader:
        start_time = time.time()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        decoded_outputs = []
        for i, output in enumerate(outputs):
            response = tokenizer.decode(output, skip_special_tokens=True)
            pattern = r"Output:\s*(\d)"
            # print(response)
            # match = re.search(pattern, output)
            # print(re.findall(pattern, response))
            matches = re.findall(pattern, response)[1].strip() if re.findall(pattern, response) else 'nan'
            current_data = df.iloc[global_index][['d', 'abstract', 'ref_abstract']].copy()
            current_data['matches'] = matches
            decoded_outputs.append(current_data)
            global_index += 1

        batch_df = pd.DataFrame(decoded_outputs)
        results_df = pd.concat([results_df, batch_df], ignore_index=True)
        elapsed_time = time.time() - start_time
        print(f"Batch processing time: {elapsed_time}s")
        results_df.to_csv(f'data/hard_attention_{args.data}.csv', mode='a', header=False, sep='\t')
        results_df = pd.DataFrame()

        # python -u chatglm_train/model_glm/hard_atten.py --phase chatglm-32k
