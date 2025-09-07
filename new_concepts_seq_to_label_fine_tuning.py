import json
import random
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
from sklearn.metrics.pairwise import cosine_similarity
#from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import pandas as pd
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from peft import PeftConfig, PeftModel
import transformers
import gc
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["TOKENIZERS_PARALLELISM"]="false"
import torch
import torch.nn as nn
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
import numpy as np
#from evaluation import all_metrics
from tqdm import tqdm
import time
from datasets import Dataset
from sklearn.model_selection import train_test_split

from rare_50 import rare_50
from codes_50 import codes_50

print(f"PID => {os.getpid()}")

print(torch.cuda.device_count())  # Check number of available GPUs
print(torch.cuda.get_device_name(0))  # Get name of first GPU


set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

login("")

model_id = "ContactDoctor/Bio-Medical-Llama-3-8B" #"meta-llama/Meta-Llama-3-8B-Instruct" #ContactDoctor/Bio-Medical-Llama-3-8B

def create_dynamic_prompt(note, concept_preds, description, flag, tokenizer):
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
    base_info = f"The patient in the note contains {description}." if flag else f"The patient in the note does not contain {description}."
    pre_prompt = f"""
You are a medical expert tasked with evaluating the presence of a disease in a patient's medical note. Provide your answer in a clear Yes/No format, on a new line.
---
### Medical Note:"""
    note = f"""
(M) {note}"""
    ques_prompt = f"""
{base_info}
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




data_file = "./mimic3-50l_train_raw.json"

code_list = rare_50[:]


assert len(code_list) == 50

label_list = sorted(list(code_list))
num_labels = len(label_list)

label_to_id = {v: i for i, v in enumerate(label_list)}


description_file = "./mimic_full_codes_descriptions.json"
with open(description_file, 'r') as json_file:
    label_descriptions = json.load(json_file)


descriptions_to_label = {}

for l in label_list:
    descriptions_to_label[label_descriptions[l]] = l


concepts_layer1_file = "./concepts_layer1_filtered.json"
with open(concepts_layer1_file, 'r') as json_file:
    concepts_layer1 = json.load(json_file)

concepts_layer2_file = "./concepts_layer2_filtered.json"
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
    cons = concept_dict[k]
    for c in cons:
        if c not in all_concepts:
            all_concepts.append(c)

print(f"number of concepts : {len(all_concepts)}")

#while True:
#    s = 10

for l in label_list:
    if l not in concept_dict.keys():
        print(f"not in keys : {l} desc : {label_descriptions[l]}")

#output_file = "./mimic_full_concept_predictions_50pcnt.json"

concept_predictions = {}

#llm = pipeline("text-generation", model=model_id, device=device)  # Replace with your desired LLM pipeline or API
# df = pd.read_csv(data_file)
with open(data_file) as f:
    df = json.load(f)
total = 0
correct = 0

max_len = 0
curr = 0
c_1 = 0
tot_1 = 0
#df = df.sample(n=100, random_state=42)
data_list = []


print("Before operation df iteration rows:")
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

model_checkpoint = "ContactDoctor/Bio-Medical-Llama-3-8B"#"meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, padding_side = "left")
tokenizer.pad_token = tokenizer.eos_token

#for index, row in df.iterrows():

nos = 0
pos = 0
ind = -1
for index, row in tqdm(enumerate(df), desc="Processing rows"):
    curr += 1
    medical_note = row["TEXT"]
    hadm_id = str(row["hadm_id"])
    if not pd.isna(row["LABELS"]):
        labels = set([c for c in row["LABELS"].split(";") if c in label_list])
        not_in_labels = [s for s in label_list if (s not in labels)]
        option_labels = list(labels)
        st = time.time()
        
        #print(f"option labels len {len(option_labels)} labels len {len(labels)}")

        #break
        
        #c_1 += len([l for l in labels if l in option_labels])
        #tot_1 += len(option_labels)

        extras = random.sample(not_in_labels, 2)
        option_labels.extend(extras)

        c = len([l for l in labels if l in option_labels])
        t = len(option_labels)
        
        #print(f"c : {c}, t: {t}")


        total += t
        correct += c

    
        #print(f"options : {option_labels}")
        query_objects = {}
        for code in option_labels:
            desc = label_descriptions[code]
            if desc != "":
                con_pairs = []
                cons = random.sample(concept_dict[code], 5)
                for con in cons:
                    if code in labels:
                        con_pairs.append((con, "Yes"))
                    else:
                        con_pairs.append((con, "No"))
                other_labels = random.sample(not_in_labels, min(3, len(not_in_labels))) if code in labels else random.sample(list(labels), min(3, len(labels)))
                for l in other_labels:
                    con = random.sample(concept_dict[l], 1)[0]
                    if l in labels:
                        con_pairs.append((con, "Yes"))
                    else:
                        con_pairs.append((con, "No"))
                random.shuffle(con_pairs)
                query_objects[code] = con_pairs


        for k, v in query_objects.items():
            desc = label_descriptions[k]
            prompt = create_dynamic_prompt(medical_note, v, desc, (k in labels), tokenizer)
            
            if k in labels:
                pos += 1
                prompt += f" Yes"
            else:
                prompt += f" No"
                nos += 1
            
            if k in labels:
                ind = len(data_list)
            #print(f"id >> {hadm_id}, prompt >> {prompt}")

            data_list.append(prompt)
    #if curr > 20:
    #    break

print(f"{pos} vs {nos}")
print(data_list[ind])
print(data_list[-5])

print(f'###################### before inf loop ########################')

#while True:
#    s =10
# After operation

print("After operation df iterations:")
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")


#print(f"percentage of c_1 {c_1 / tot_1}")
#print(f"percentage of correct {correct / total}")
#print(f"count per note {total / len(df)}")
#print(f"data list len : {len(data_list)}")


#while True:
#    s=10

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f'Found max lenth: {max_length}')
            break
    if not max_length:
        max_length = 8192
        print(f'Using default max length: {max_length}')
    return max_length


print("Before operation model and tokenizer initiation:")
print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

"""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
"""

bnb_config = BitsAndBytesConfig(
    #load_in_4bit=False,
    load_in_8bit=True,
    #bnb_8bit_compute_dtype=torch.bfloat16
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Define your model checkpoint
model_checkpoint = "ContactDoctor/Bio-Medical-Llama-3-8B"#"meta-llama/Meta-Llama-3-8B"

# Load tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    #quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map={"":0},
)

model.gradient_checkpointing_enable()


def print_trainable_parameters(model):
    """
    printing the number of trainable paramters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


config = LoraConfig(
    r = 32,
    lora_alpha = 128,
    target_modules = ["k_proj", "v_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

# Prepare dataset
dataset = Dataset.from_dict({"text": data_list})


dataset = dataset.map(lambda samples: tokenizer(samples['text'], truncation=True, padding=True), batched=True)
dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < 8192)


trainer = transformers.Trainer(
    model = lora_model,
    train_dataset = dataset,
    args = transformers.TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        report_to="none",
        warmup_steps = 2,
        max_steps = 500,
        learning_rate = 2e-4,
        #fp16 = True,
        logging_steps = 1,
        output_dir = './new_rare_ft_concepts_seq_to_label_output',
        optim = "paged_adamw_8bit",
        save_strategy = "steps",
        save_steps=50
    ),
    data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False)
)
model.config.use_cache = False
trainer.train()

trainer.model.save_pretrained("./new_rare_ft_concepts_seq_to_label")
tokenizer.save_pretrained("./new_rare_ft_concepts_seq_to_label")
print("Fine-tuning completed and model saved.")

