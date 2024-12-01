# import os
# os.environ[/"CUDA_VISIBLE_DEVICES"] = "1,3"
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,AutoModelForCausalLM,AutoModel
from tqdm import tqdm
from prompt import *
import random
import os
from data_loader import DBLPDataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from model.chatglm_adapter import chatglm_adapter
from model.chatglm_mlp import chatglm_mlp
from model.RAHA import ChatGLM_TEA
from torch.cuda.amp import autocast, GradScaler
from torch.nn import MSELoss, L1Loss
import ast
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data.distributed import DistributedSampler

# Initialize grad scaler for mixed precision training
scaler = GradScaler()

# Function to create data loaders for the dataset
def create_data_loader(grouped_df, tokenizer, batch_size):
    dataset = DBLPDataset(grouped_df, tokenizer)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader


# Function to format and create prompts from differences
def difference_generation(differences):
    formatted_prompts = [f"No.{id + 1} difference: {diff}" for id, diff in enumerate(differences)]
    return ' '.join(formatted_prompts)


def update_prompts(df):
    df['input'] = df.apply(
        lambda row: prompt_generation(row['abstract'], row['difference'], row.get('predictions', '')), axis=1)
    return df


# Function to train the model
def train_model(model, loss_function, epochs, train_data, tokenizer, initial_batch_size, device, update_prompts_flag,
                data, ori,adapter, val_data):
    best_mae = float('inf')
    best_model_dir = 'result_model'
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)
    best_model_path = f'result_model/{data}_{update_prompts_flag}_{ori}_{adapter}_best.pth'
    if os.path.exists(best_model_path):
        print(f"Loading weights from the best model at {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("No pre-trained best model found, starting training from scratch.")

    for epoch in range(epochs):
        model.train()
        model.float()
        total_loss = 0
        total_mae = 0
        predictions_list = []
        train_loader = create_data_loader(train_data, tokenizer, initial_batch_size)
        optimizer = torch.optim.Adam(model.parameters())

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), \
            batch['label'].to(device)
            optimizer.zero_grad()
            predictions,_ = model(input_ids=input_ids, attention_mask=attention_mask, generate_text=False)
            predictions = predictions.squeeze(-1)
            loss = loss_function(predictions, labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()
            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions - labels)).item()
            predictions_list.extend(predictions.tolist())

        if update_prompts_flag:
            if 'predictions' not in train_data.columns:
                train_data['predictions'] = np.nan
            train_data['predictions'][0:len(predictions_list)] = predictions_list
            update_prompts(train_data)

        avg_loss, avg_mae = total_loss / len(train_loader), total_mae / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss (MSE): {avg_loss}, MAE: {avg_mae}")

        avg_val_mae = validate_model(model, loss_function, 1, val_data, tokenizer, 4, device, update_prompts_flag, data,
                                     ori, adapter)
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            torch.save(model.state_dict(), best_model_path)

# Validation function to evaluate model performance
def validate_model(model, loss_function, epochs, val_data, tokenizer, batch_size, device, update_prompts_flag, data,
                   ori, adapter):
    model.eval()
    model.float()
    total_val_loss, total_mae, predictions_list = 0, 0, []
    val_loader = create_data_loader(val_data, tokenizer, batch_size)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), \
            batch['label'].to(device)
            predictions,_ = model(input_ids=input_ids, attention_mask=attention_mask, generate_text=True)
            predictions = predictions.squeeze(-1)
            loss = loss_function(predictions, labels)
            total_val_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions - labels)).item()
            predictions_list.extend(predictions.cpu().numpy().tolist())

        if update_prompts_flag:
            if 'predictions' not in val_data.columns:
                val_data['predictions'] = np.nan
            val_data['predictions'][0:len(predictions_list)] = predictions_list
            update_prompts(val_data)

    avg_val_loss, avg_mae = total_val_loss / len(val_loader), total_mae / len(val_loader)
    print(f"Validation Loss (MSE): {avg_val_loss}, MAE: {avg_mae}")
    return avg_mae


# Test function to evaluate the model on a test dataset
def test_model(model, loss_function, epochs, test_data, tokenizer, batch_size, device, update_prompts_flag, data, ori,
               adapter):
    model.eval()
    model.float()
    total_test_loss, total_mae, predictions_list = 0, 0, []
    test_loader = create_data_loader(test_data, tokenizer, batch_size)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids, attention_mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), \
            batch['label'].to(device)
            predictions,_ = model(input_ids=input_ids, attention_attack=attention_mask, generate_text=True)
            predictions = predictions.squeeze(-1)
            loss = loss_function(predictions, labels)
            total_test_loss += loss.item()
            total_mae += torch.mean(torch.abs(predictions - labels)).item()
            predictions_list.extend(predictions.cpu().numpy().tolist())

        if update_prompts_flag:
            if 'predictions' not in test_data.columns:
                test_data['predictions'] = np.nan
            test_data['predictions'][0:len(predictions_list)] = predictions_list
            update_prompts(test_data)

    avg_test_loss, avg_mae = total_test_loss / len(test_loader), total_mae / len(test_loader)
    print(f"Test Loss (MSE): {avg_test_loss}, MAE: {avg_mae}")
    return avg_mae


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GLM model for generating summary.")
    parser.add_argument('--phase', type=str, help="Phase to control which operation to perform (train/validate)",
                        required=True)
    parser.add_argument('--model', type=str, help="Model", required=True)
    parser.add_argument('--epoch', type=int, help="Batch size for training and validation", default=4)
    parser.add_argument('--data', type=str, help="Path to the data file", default='diff_gene_pmc.csv')
    parser.add_argument('--update', action='store_true', help="Whether to update prompts after predictions")
    parser.add_argument('--ori', action='store_true', help="Whether to ori")
    parser.add_argument('--adapter', action='store_true', help="Whether to adapter")
    args = parser.parse_args()

    # Setup for distributed training across GPUs if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model selection based on user's input argument
    if args.model in ['chatglm3-6b-32k', 'bloom', 'Meta-Llama-3-8B']:
        model_name = args.model
        print(model_name)
        model = AutoModelForCausalLM.from_pretrained(f"{model_name}", trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}", trust_remote_code=True)

    elif args.model == 'bert':
        model_name = args.model
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        # model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

    elif args.model == 't5':
        model_name = args.model
        tokenizer = T5Tokenizer.from_pretrained('t5')
        # model = T5ForConditionalGeneration.from_pretrained('t5')

    elif args.model == 'roberta':
        model_name = args.model
        tokenizer = RobertaTokenizer.from_pretrained('roberta')
        # model = RobertaModel.from_pretrained('roberta')

    df = pd.read_csv(f'data/{args.data}_info.csv', sep='\t', header=None,
                     names=['index', 'abstract', 'reference', 'd', 'p', 'atten', 'input_len','difference'])

    if args.ori:
        grouped_df = df.groupby(['abstract', 'd'])['difference'].apply(list).reset_index(name='difference')
        grouped_df['input'] = grouped_df.apply(lambda row: difference_generation(row['difference']), axis=1)

    else:
        df = df[df['atten'] == 1]
        grouped_df = df.groupby(['abstract', 'd'])['difference'].apply(list).reset_index(name='difference')
        grouped_df['input'] = grouped_df.apply(lambda row: difference_generation(row['difference']), axis=1)

    train_data, temp_data = train_test_split(grouped_df, test_size=0.7, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    save_directory = 'split'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    train_data.to_csv(f'split/{args.data}_train.csv', index=False)
    val_data.to_csv(f'split/{args.data}_val.csv', index=False)
    test_data.to_csv(f'split/{args.data}_test.csv', index=False)
    print(f"Train data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")
    print(f"Test data size: {len(test_data)}")

    # Save the preprocessed data and start the model training or testing process based on the phase argument
    if args.adapter:
        model = ChatGLM_TEA(model_name).to(device)
    else:
        model = chatglm_mlp(model_name).to(device)

    # Based on the 'phase' flag, train or test the model
    if args.phase == 'train':
        train_data_path = f'split/{args.data}_train.csv'
        val_data_path = f'split/{args.data}_val.csv'
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        train_model(model, nn.MSELoss(), args.epoch, train_data, tokenizer, 4, device, args.update, args.data, args.ori,
                    args.adapter, val_data)
    elif args.phase == 'test':
        test_data_path = f'split/{args.data}_test.csv'
        test_data = pd.read_csv(test_data_path)
        best_model_path = f'result_model/{args.data}_{args.update}_{args.ori}_{args.adapter}_best.pth'
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_model(model, nn.MSELoss(), args.epoch, test_data, tokenizer, 4, device, args.update, args.data, args.ori,
                   args.adapter)