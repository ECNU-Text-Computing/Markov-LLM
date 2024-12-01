import torch
import re
import pandas as pd
from data_loader import DBLPDataset
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from prompt import prompt_difference
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run GLM model for generating differences.")
    parser.add_argument('--phase', type=str, required=True, help="Phase to control which model to use")
    parser.add_argument('--data', type=str, required=True, help="Phase to control which data to use")
    args = parser.parse_args()

    # Set up device for model computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize tokenizer and model based on the specified phase
    model_map = {
        'chatglm': "chatglm3-6b",
        'chatglm3-6b-32k': "chatglm3-6b-32k",
        'chatglm-base': "chatglm3-6b-base"
    }
    if args.phase in model_map:
        model_name = model_map[args.phase]
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        raise ValueError("Unsupported phase argument")

    # Load data and filter by attention value
    df = pd.read_csv(f'data/hard_attention_{args.data}.csv', sep='\t', header=None,
                       names=['index', 'd', 'abstract', 'reference', 'atten'])

    # Prepare the input by appending prompts
    print("Begin to generate difference!!!")
    df.dropna(subset=['abstract', 'reference'], inplace=True)
    df['input'] = df.apply(lambda row: prompt_difference(row['abstract'], row['reference']), axis=1)
    df['input_length'] = df['input'].apply(len)
    df = df[['abstract', 'reference', 'd', 'input', 'atten', 'input_length']].reset_index(drop=True)

    # Sort data by input length in descending order
    df_sorted = df.sort_values(by='input_length', ascending=False).reset_index(drop=True)
    print(df_sorted.head())

    # Initialize data loader
    dataset = DBLPDataset(df_sorted, tokenizer)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    tqdm(loader, desc='Loading data')

    # Load model, set to evaluation mode, and define generation parameters
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()
    gen_kwargs = {
        "max_length": 8192,
        "num_beams": 1,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8
    }

    results_df = pd.DataFrame()
    global_index = 0
    for batch in tqdm(loader, desc='Generating text'):
        start_time = time.time()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model.generate(input_ids, attention_mask=attention_mask, **gen_kwargs)

        decoded_outputs = []
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            pattern = r"Contrast and Difference:\s*(.*?)(?=\n|$)"
            matches = re.search(pattern, response).group(1) if re.search(pattern, response) else 'nan'
            # print(matches)
            current_data = df_sorted.iloc[global_index].copy()
            current_data['matches'] = matches
            decoded_outputs.append(current_data.to_dict())
            global_index += 1

        batch_df = pd.DataFrame(decoded_outputs)
        results_df = pd.concat([results_df, batch_df], ignore_index=True)

        elapsed_time = time.time() - start_time
        print(f"Batch processing time: {elapsed_time}s")

    # Save results to a CSV file
    results_df.to_csv(f'data/{args.data}_info.csv', mode='a', header=False, sep='\t')

if __name__ == '__main__':
    main()