import json
import random
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import pandas as pd
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel
import transformers
import gc
import os
import sys
import re
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)
import math
import torch
import torch.nn as nn
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
import numpy as np
from evaluation import all_metrics
from tqdm import tqdm
import time
from datasets import Dataset
from sklearn.model_selection import train_test_split

from codes_50 import codes_50
from rare_50 import rare_50

print(f"PID => {os.getpid()}")

print(torch.cuda.device_count())  # Check number of available GPUs
print(torch.cuda.get_device_name(0))  # Get name of first GPU


set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

login("")


def create_dynamic_prompt(note, concept_preds, description, tokenizer):
    """
    Generates a structured prompt dynamically for a given medical note, set of concepts, and description.

    Parameters:
    - note (str): The medical note to evaluate.
    - concepts (list): A list of concepts (C = [c1, c2, c3, ..., cn]).
    - description (str): The overall description D.

    Returns:
    - str: The generated prompt.
    """

    # Build the full prompt
    prev_qas = "\n".join([f"Question: Does the patient contain '{concept}'?\nAnswer: {pred}" for (concept, pred) in concept_preds])
    #base_info = f"The patient in the note contains {description}." if flag else f"The patient in the note does not contain {description}."
    pre_prompt = f"""
You are a medical expert tasked with evaluating the presence of a disease in a patient's medical note. Provide your answer in a clear Yes/No format, on a new line.
---
### Medical Note:"""
    note = f"""
(M) {note}"""
    ques_prompt = f"""
Previously you were asked about the presence of some medical concepts in the note (M). The questions, and your answers to each of them are provided below:
{prev_qas}
Based on your own answers to the above questions, answer the following question regarding the presence of a disease in the note (M).
---
### Question:
(Q) Based on the medical note (M), and your answers to the previous questions, does the patient contain the disease '{description}'?

Answer the question above in a clear Yes/No format, on a new line.
---

### Output Format:
- Answer each question on a new line in the format:
  'Yes' or 'No'

### Response:"""

    pretext_tokens = tokenizer(pre_prompt, return_tensors="pt")['input_ids'].size(1)
    qa_tokens = tokenizer(ques_prompt, return_tensors="pt")['input_ids'].size(1)
    remaining_tokens = 8192 - pretext_tokens - qa_tokens

    # Truncate the note if needed
    if remaining_tokens > 0:
        truncated_note = tokenizer.decode(
            tokenizer(note, max_length=remaining_tokens, truncation=True)['input_ids'],
            skip_special_tokens=True
        )
    else:
        truncated_note = ""  # If no space remains, discard the note
        
    prompt = f"{pre_prompt}\n{truncated_note}\n{ques_prompt}"
    return prompt

data_file = "../test_50.csv" # your file for rare-50

#code_file = codes_50 # rare_50"../ALL_CODES_50.txt"

codes = codes_50[:] # rare_50"
"""
with open(code_file, "r") as f:
    for line in f:
        if line.strip() != "":
            codes.add(line.strip())
"""
assert len(codes) == 50

label_list = sorted(list(codes))
num_labels = len(label_list)

label_to_id = {v: i for i, v in enumerate(label_list)}


description_file = "../mimic_full_codes_descriptions.json"
with open(description_file, 'r') as json_file:
    label_descriptions = json.load(json_file)


descriptions_to_label = {}

for l in label_list:
    descriptions_to_label[label_descriptions[l]] = l


concepts_layer1_file = "../concepts_layer1_filtered.json"
with open(concepts_layer1_file, 'r') as json_file:
    concepts_layer1 = json.load(json_file)

concepts_layer2_file = "../concepts_layer2_filtered.json"
with open(concepts_layer2_file, 'r') as json_file:
    concepts_layer2 = json.load(json_file)

concept_dict = {}
for obj in concepts_layer1:
    k = obj["code"]
    v = obj["concept_1"]
    t = []
    for c in v.split(";"):
        if c not in t:
            t.append(c)
    concept_dict[k] = t

for obj in concepts_layer2:
    k = obj["code"]
    v = obj["concept_2"]
    t = concept_dict[k]
    for c in v.split(";"):
        if c not in t:
            t.append(c)
    concept_dict[k] = t

print(f"concept - codes : {len(concepts_layer1)}, {len(concepts_layer2)}") 

#for k in concept_dict.keys():
#    concept_dict[k] = set(v)

all_concepts = []
for k in concept_dict.keys():
    if k in label_list:
        cons = concept_dict[k]
        for c in cons:
            if c not in all_concepts:
                all_concepts.append(c)


all_concepts = sorted(all_concepts)
print(f"number of concepts : {len(all_concepts)}")


for l in label_list:
    if l not in concept_dict.keys():
        print(f"not in keys : {l} desc : {label_descriptions[l]}")

concept_predictions = {}

df = pd.read_csv(data_file) # or, handle for json file types
total = 0
correct = 0
all_gt = np.zeros((len(df), num_labels))
all_preds = np.zeros((len(df), num_labels))
all_preds_raw = np.zeros((len(df), num_labels))

max_len = 0
curr = 0
c_1 = 0
tot_1 = 0
#df = df.sample(n=100, random_state=42)


print("Before operation df iteration rows:")
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

PEFT_MODEL = "./new_ft_concepts_seq_to_label/"#"../new_ft_output/checkpoint-1050/"#take from hugg 

config = PeftConfig.from_pretrained(PEFT_MODEL)

peft_base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    #return_dict=True,
    #quantization_config=bnb_config,
    #device_map="auto",
    device_map={"":0},
    #trust_remote_code=True,
)

peft_model = PeftModel.from_pretrained(peft_base_model, PEFT_MODEL)
model = peft_model
peft_tokenizer= AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side = 'left')
peft_tokenizer.pad_token = peft_tokenizer.eos_token
tokenizer = peft_tokenizer
#for index, row in df.iterrows():

with open("./mimic3_50_concepts_binary_preds.json", 'r') as json_file: # concept prediction file
    preds_dict = json.load(json_file)

ind = -1


#import torch
#from transformers import LogitsProcessorList

def get_token_logprobs(model, tokenizer, input_text, max_new_tokens=2):
    """
    Generate `max_new_tokens` tokens and extract their log probabilities.
    """
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # Generate outputs with log probabilities
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,  # Enables log probabilities
        )

    # Extract log probabilities
    generated_tokens = outputs.sequences[:, inputs.input_ids.shape[1]:]  # Only new tokens
    token_scores = outputs.scores  # List of logit tensors (one per token generated)

    # Convert logits to log probabilities
    log_probs = []
    for i, logits in enumerate(token_scores):
        probs = torch.log_softmax(logits, dim=-1)  # Convert logits to log probabilities
        token_id = generated_tokens[0, i].item()  # Extract token ID
        log_probs.append(probs[0, token_id].item())  # Store log probability

    return generated_tokens[0], log_probs  # Return token IDs & log probabilities

def get_highest_logprob(model, tokenizer, input_text):
    """
    Generate 2 tokens and find the highest log probability among "Yes", "yes", "YES".
    """
    # Generate tokens and log probabilities
    generated_tokens, log_probs = get_token_logprobs(model, tokenizer, input_text)

    # Decode generated tokens
    generated_texts = [tokenizer.decode([tok]).strip() for tok in generated_tokens]

    # Get log probabilities for "Yes", "yes", "YES"
    yes_variants = ["Yes", "yes", "YES"]
    yes_logprobs = [log_probs[i] for i, word in enumerate(generated_texts) if word in yes_variants]

    # Return the highest log probability among "Yes", "yes", "YES"
    l_prob = max(yes_logprobs) if yes_logprobs else None
    generated_texts = [t.lower() for t in generated_texts]
    return l_prob, generated_texts


curr = 0

for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
    curr += 1
    medical_note = row["TEXT"]
    hadm_id = str(row["HADM_ID"])
    cons_preds = preds_dict[hadm_id]
    not_in_concepts = [c for c in all_concepts if c not in cons_preds]
    labels = set([c for c in row["LABELS"].split(";") if c in label_list]) if not pd.isna(row["LABELS"]) else set()
    for idx, label in enumerate(labels):
        all_gt[index, label_to_id[label]] = 1

    for l in label_list:
        desc = label_descriptions[l]
        if desc != "":
            con_pairs = []
            cons = random.sample(concept_dict[l], 6)
            cons.extend(random.sample(not_in_concepts, 2))
            random.shuffle(cons)
            for c in cons:
                if c in cons_preds:
                    con_pairs.append((c, "Yes"))
                else:
                    con_pairs.append((c, "No"))

            prompt = create_dynamic_prompt(medical_note, con_pairs, desc, tokenizer)

            log_prob, gen_words = get_highest_logprob(model, tokenizer, prompt)
            log_prob = 0 if log_prob is None else math.exp(log_prob)
            #if l in labels:
            #    print(f"{hadm_id} > {prompt}")
            #    print(f"log_prob : {log_prob}, {gen_words}")
            all_preds_raw[index, label_to_id[l]] = log_prob
            all_preds[index, label_to_id[l]] = 1 if "yes" in gen_words else 0
    
    if (curr%20) == 0:
        metrics = all_metrics(yhat=all_preds[:curr], y=all_gt[:curr], yhat_raw=all_preds_raw[:curr])
        report = f"curr : {curr}, metrics : {metrics}\n"
        print(report)
        with open("../previous_works/top_50/metrics_concepts_seq_to_label_50.txt", "a") as f: #change the location
            f.write(report)
        np.save('../previous_works/top_50/all_preds_concepts_seq_to_label_50.npy', all_preds) #change the location
        np.save('../previous_works/top_50/all_preds_raw_concepts_seq_to_label_50.npy', all_preds_raw) #change the location


metrics = all_metrics(yhat=all_preds[:curr], y=all_gt[:curr], yhat_raw=all_preds_raw[:curr])#[:curr])
print(f"metrics {metrics}")
report = f"curr : {curr}, metrics : {metrics}\n"
with open("../previous_works/top_50/metrics_concepts_seq_to_label_50.txt", "a") as f: #change the location
    f.write(report)
np.save('../previous_works/top_50/all_preds_concepts_seq_to_label_50.npy', all_preds) #change the location
np.save('../previous_works/top_50/all_preds_raw_concepts_seq_to_label_50.npy', all_preds_raw) #change the location

