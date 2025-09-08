import json
import random

from create_prompt_concepts import create_concept_prompt
from rare_50 import rare_50
from codes_50 import codes_50

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
import os
import transformers
from peft import PeftConfig, PeftModel
import sys

concept_file = "./concepts_layer1_filtered.json" #layer1 or layer2
data_file = "mimic3-50l" #mimic3-50 or mimic3-50l
type_file = "train" #train, test or dev
concept_picking_threshold = 0.045 # 0.03 for mimic3-50l layer2, 0.045 for mimic3-50l
model_id = "meta-llama/Meta-Llama-3-8B"
model_saving_path = "./rare_model_concept_1/" # model saving path
model_output_path = "./rare_model_concept_1_output/" # model output path
concept_type = "concept_1" #concept_2 for leaf level
code_list = rare_50[:] #codes_50 for mimic3-50


with open(concept_file, "r") as f:
	codes = json.load(f)
	
# with open('../concepts_layer2_deduplicated.json', "r") as f:
# 	codes = json.load(f)
	
concept_dict = {}
for obj in codes:
    if obj['code'] in code_list:
        concept_dict[obj['code']] = obj[concept_type].split(";")

selected_titles = [
            'discharge diagnosis',
            'major surgical or invasive procedure',
            'history of present illness',
            'past medical history',
            'brief hospital course',
            'chief complaint',
            'physical exam',
            'discharge medications',
            'discharge disposition',
            'medications on admission',
            'discharge instructions',
            'followup instructions'
        ]

all_concepts = set([c for k, v in concept_dict.items() for c in v])
print(f'total number of concepts : {len(all_concepts)}')
data_list = []

#base = 'train'
# for base in ['train', 'test', 'dev']:
#t = 'mimic3-50l'#'mimic3-50', 
         
with open(f'./{data_file}_{type_file}.json') as f:
    data = json.load(f)
for d in data:
    if data_file == 'mimic3-50':
        if random.random() > 0.10:
            continue
    note = ""
    sections = d['sections']
    for k in sections.keys():
        if k in selected_titles:
            note = note + k + ": " + sections[k] + "\n"

    labels = [l for l in d['labels'].split(';') if l in code_list]

    current_concepts = []
    for l in labels:
        for c in concept_dict.get(l, []):
            if c not in current_concepts:
                current_concepts.append(c)
    
    true_concepts_index = len(current_concepts)

    current_concepts.extend([c for c in all_concepts if c not in current_concepts and random.random() < concept_picking_threshold])

    for i, c in enumerate(current_concepts):
        prompt = create_concept_prompt(note, c, i < true_concepts_index)
        data_list.append(prompt)

print(f'total data size: {len(data_list)}') 
# print(data_list[0])
no = 0
yes = 0

for d in data_list:
    if d[-2:] == 'no':
        no += 1
    else:
        yes += 1

print(f'yes : {yes}, no : {no}')

# while True:
#     s = 10

random.shuffle(data_list)
random.shuffle(data_list)

train_dataset = Dataset.from_dict({"text": data_list})

login("")
# model_id = "BioMistral/BioMistral-7B"
#model_id = "meta-llama/Meta-Llama-3-8B" # take from input
# model_id = "google/gemma-7b"
# model_id = "epfl-llm/meditron-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)




tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
# print(tokenizer.special_tokens_map)
tokenizer.pad_token = tokenizer.eos_token


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    #print(f't params: {trainable_params}, all params: {all_param}, t%: {100 * trainable_params / all_param}')

def get_max_length(model):
    conf = model.config
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f'Found max lenth: {max_length}')
            break
    if not max_length:
        max_length = 1024
        print(f'Using default max length: {max_length}')
    return max_length



model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)


config = LoraConfig(
    r=32,#8
    lora_alpha=32,
    # target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
# print_trainable_parameters(model)
# 85041152 || all params: 3837112320 || trainable%: 2.2162799758751914 : Biomistral - 7b


max_length = get_max_length(model) if get_max_length(model) is not None else 8192

train_dataset = train_dataset.map(lambda samples: tokenizer(samples["text"], truncation=True, padding=True), batched=True)
train_dataset = train_dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)


print('checking data train : ')
print(train_dataset['text'][0])


training_args = transformers.TrainingArguments(
        report_to="none",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=100,#100
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=model_output_path,
        optim="paged_adamw_8bit",
        # logging_dir="/content/gdrive/MyDrive/MaLab/MIMIC/Outputs/",
        save_strategy="epoch",
        # save_steps=10,
        # evaluation_strategy="steps",
        # eval_steps=100,
        # do_eval=True
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    # train_dataset=data,
    #eval_dataset=dataset_dev,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

print('training done.')
model_to_save = trainer.model.module if hasattr(trainer.model, 'module') else trainer.model  # Take care of distributed/parallel training
model_to_save.save_pretrained(model_saving_path)
trainer.save_model(model_saving_path)
#trainer.push_to_hub()
print('saving done')
