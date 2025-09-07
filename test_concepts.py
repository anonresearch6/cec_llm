import json
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel
from datasets import Dataset, load_dataset
import os
import transformers
import sys

from create_prompt_concepts import create_concept_prompt
#########    Supporting Information    ##########

concept_file = "" #layer1 or layer2
data_file = "" #mimic3-50 or mimic3-50l
type_file = "" #train, test or dev
concept_picking_threshold = 0.045 # 0.03 for mimic3-50, 0.045 for mimic3-50l
model_saving_path = "" # model saving path
model_output_path = "" # model output path

with open('./concepts_layer1_deduplicated.json', "r") as f:
	codes = json.load(f)

concept_dict = {}
for obj in codes:
	concept_dict[obj['code']] = obj['concept_1'].split(";")

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

# login("")
# model_id = "BioMistral/BioMistral-7B"
model_id = "meta-llama/Meta-Llama-3-8B" # take from input
# model_id = "google/gemma-7b"
# model_id = "epfl-llm/meditron-7b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

PEFT_MODEL = 'trained_model_location'

config = PeftConfig.from_pretrained(PEFT_MODEL)

peft_base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    #return_dict=True,
    quantization_config=bnb_config,
    #device_map="auto",
    device_map={"":0},
    #trust_remote_code=True,
)

peft_model = PeftModel.from_pretrained(peft_base_model, PEFT_MODEL)
model = peft_model 
peft_tokenizer= AutoTokenizer.from_pretrained(config.base_model_name_or_path, padding_side = 'left')
peft_tokenizer.pad_token = peft_tokenizer.eos_token
tokenizer = peft_tokenizer
model.config.use_cache = False


#########    INFERENCE    ##########



#dataset_test = dataset['test'].filter(lambda example: example['run'] <= 5)
#print(dataset)
#dataset_test = dataset_test.map(lambda samples: tokenizer(samples["text"], truncation=True, padding=True), batched=True)
device = "cuda:0"
#dataset_test2 = tokenizer(dataset_test['text'], padding=True, truncation=True, max_length=8192, return_tensors="pt").to(device)
#assert dataset_test2['input_ids'].shape[0] == len(dataset_test)
#device = "cuda:0"
i = 0
correct = 0
cnt = 0
predicted_list = []
k = 2
#k = 4
print('starting inference')

base = 'train'
# for base in ['train', 'test', 'dev']:
t = 'mimic3-50l'#'mimic3-50', 
         
with open(f'./{t}_{base}.json') as f:
    data = json.load(f)

all_preds = {}
with torch.no_grad():
    print("inside torch.nograd")
    for index, d in enumerate(data):
        note = ""
        sections = d['sections']
        for k in sections.keys():
            if k in selected_titles:
                note = note + k + ": " + sections[k] + "\n"

        labels = d['labels'].split(';')
        hadm_id = d['hadm_id']
        current_preds = []
        for i, c in enumerate(all_concepts):
            prompt = create_concept_prompt(note, c, flag=(c in labels), train=False)
            encoding = tokenizer(prompt, padding=True, return_tensors='pt').to(device)#
            generated_ids = model.generate(**encoding, max_new_tokens=2)#
            #for encoding in encodings:
                
            #generated_ids = model.generate(encodings, max_new_tokens=2)
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            pred = generated_text[len(prompt):].strip().lower()
            if 'yes' in pred and c not in current_preds:
                 current_preds.append(c)
        all_preds[hadm_id] = current_preds

        if index % 50 == 0:
            with open('icd_coding/data/concepts_2/llama3_main/output/prediction_temp.json', 'w') as f:			
                json.dump(all_preds, f)
        
with open('icd_coding/data/concepts_2/llama3_main/output/prediction_temp.json', 'w') as f:
    json.dump(all_preds, f)

# print(generated_texts[0])
# better
#texts = 
#input_text = texts[5] # i
# print(input_text)
# label = test_labels[i].lower().strip() # i
# label = [test_label.lower() for test_label in test_labels[i:i+4]] # i
# print(test_labels[5])
# print(input_text)
# device = "cuda:0"
# inputs = tokenizer(input_text, return_tensors="pt").to(device)

#inputs = tokenizer(input_text, return_tensors="pt").to(device)
#outputs = model.generate(**inputs, max_new_tokens=2) #**
#outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)#.split(':')[-1].strip().lower()
# print(f'acc : {correct/cnt}')


