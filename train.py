import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
import pandas as pd
import os
from datasets import Dataset
import numpy as np
import argparse
import csv
from transformers import TrainingArguments, Trainer


def get_product_details(cod_product):
    try:            
        product_row = product_csv.loc[product_csv['cod_modelo_color'] == cod_product]

        characteristics = {
            "product": product_row['des_product_type'].tolist()[0],
            "color": product_row['des_agrup_color_eng'].tolist()[0],
            "material": product_row['des_fabric'].tolist()[0]
            }

        return characteristics
    except Exception:
        return None

def find_products_from_outfit(cod_outfit_find, outfit_csv):
    try:
        outfit = outfit_csv.loc[outfit_csv['cod_outfit'] == cod_outfit_find, 'cod_modelo_color'].tolist()
        
        if not outfit:
            return None
        
        return outfit
    except Exception:
        pass


def get_outfit_details(outfit_list):
    product_details = []
    for prod in outfit_list:
        details = get_product_details(prod)

        if details != None:
            product_details.append(details)

    return product_details

def get_products_outfit(examples, product_csv, outfit_csv):
    return_list = []
    
    for outfit in examples["outfit_idx"]:
        
        outfit_products = find_products_from_outfit(outfit, outfit_csv)
        
        return_list.append(outfit_products)
        
    return {"outfit_products": return_list}
    
def gen_outfit_prompt(outfit_det, prefix):
    txt_string = prefix
    for product in outfit_det:
        txt_string += f'- Product: {product["product"]}, Color: {product["color"]}, Material: {product["material"]}\n'
    return txt_string 

def generate_training_prompt(examples, prefixes):
    return_text = []
    
    a = examples["outfit_idx"]
    return_indx = [val for val in a for _ in range(len(prefixes))]

    for outfit in examples["outfit_products"]:
        for prefix in prefixes:
            juan = get_outfit_details(outfit)
            return_text.append(gen_outfit_prompt(juan, prefix))

    return {"outfit_idx": return_indx, "text": return_text}

def tokenize_dataset(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)

def process_dataset(product_csv, outfit_csv, tokenizer, prefixes):
    outfit_idx_list = list(outfit_csv["cod_outfit"].unique())
    idx_ds = Dataset.from_list([{"outfit_idx": sample} for sample in outfit_idx_list])#.select(range(5))
    outfit_ds = idx_ds.map(get_products_outfit, fn_kwargs= {"product_csv": product_csv, "outfit_csv": outfit_csv}, batched = True)
    outfit_ds = outfit_ds.map(generate_training_prompt, fn_kwargs= {"prefixes": prefixes}, batched = True, remove_columns = ['outfit_idx', 'outfit_products'])
    outfit_ds = outfit_ds.map(tokenize_dataset, fn_kwargs= {"tokenizer": tokenizer}, batched = True, remove_columns = ['outfit_idx', 'text'])
    return outfit_ds


def train(args):

    directorio_cwd = os.getcwd()
    directorio_product = os.path.join(directorio_cwd, "datathon", "dataset", "filtered_product_data.csv")
    directorio_outfit = os.path.join(directorio_cwd, "datathon", "dataset", "outfit_data.csv")

    product_csv = pd.read_csv(directorio_product)
    outfit_csv = pd.read_csv(directorio_outfit)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "tj-solergibert/Mango-DA-GPT2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    prefixes = ["The outfit is composed of:\n", "The outfit it made up of:\n", "The outfit is formed by:\n"]
    
    ds = process_dataset(product_csv, outfit_csv, tokenizer, prefixes).train_test_split(test_size=0.1)

    training_args = TrainingArguments(
    output_dir="./resultsDAAA",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=args.batch_size,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    )

    trainer.train()

    model_id = "Mango-DA-GPT2-B"
    model_path = os.path.join(directorio_cwd, model_id)
    tokenizer.save_pretrained(model_path)
    trainer.model.save_pretrained(model_path)

    if args.push_to_hub:
        hf_model_id = f"tj-solergibert/{model_id}"
        trainer.model.push_to_hub(hf_model_id)
        tokenizer.push_to_hub(hf_model_id)

def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to upload the trained model to HuggingFace",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch Size",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()