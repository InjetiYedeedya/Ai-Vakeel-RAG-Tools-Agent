# Generated from: Ai_vakeel_Fine_tunning.ipynb
# Converted at: 2026-03-30T18:26:00.764Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# ### 1. Load & Convert PDF to Text


!pip install pypdf

from pypdf import PdfReader
import os

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

text = ""

file_path = "data.txt"

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
else:
    folder_path = "data"
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            text += extract_text_from_pdf(pdf_path)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

print(text[:1000])

# ### 2. Convert Text → Dataset


from datasets import Dataset

def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = split_text(text)

dataset = Dataset.from_dict({"text": chunks})

# ### 3. Load Model + Tokenizer


from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'distilgpt2' # lightweight

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto') # uses gpu

# ### 4. Apply LoRA


from peft import LoraConfig, get_peft_model


from peft import LoraConfig, get_peft_model
lora_config = LoraConfig(
    r = 4,
    lora_alpha = 16,
    target_modules = ['c_attn'],
    lora_dropout=0.1,
    bias='none',
    task_type = 'CAUSAL_LM'
)

model = get_peft_model(model,lora_config)

# ### 5. Tokenization


def tokenize_function(example):
    return tokenizer(
        example['text'],
        padding='max_length',
        truncation = True,
        max_length=256
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ### 6. Training


tokenized_dataset = tokenized_dataset.map(
    lambda x: {
        "input_ids": x["input_ids"],
        "attention_mask": x["attention_mask"],
        "labels": x["input_ids"]
    },
    remove_columns=tokenized_dataset.column_names
)

model.save_pretrained("lora-law-model")
tokenizer.save_pretrained("lora-law-model")

model = model.merge_and_unload()   # 🔥 FIX

from transformers import pipeline

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "What is Indian Penal Code?"

print(pipe(prompt, max_length=200)[0]["generated_text"])

# ### 7. QloRA


from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token



!pip install -U transformers accelerate peft bitsandbytes datasets

import torch
import bitsandbytes as bnb

print("GPU:", torch.cuda.is_available())
print("bitsandbytes:", bnb.__version__)

import torch
print(torch.cuda.is_available())

!nvidia-smi

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)

from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # 🔥 for phi-2
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./qlora-law",
    per_device_train_batch_size=1, # batch size
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    fp16=True,
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()

from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

prompt = "Question: What is Indian Penal Code?\nAnswer:"

print(pipe(prompt, max_length=200)[0]["generated_text"])



# ### 10. Modified code


!pip install -U transformers accelerate peft bitsandbytes datasets pypdf -q

from pypdf import PdfReader
import os

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

text = ""

file_path = "data.txt"

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
else:
    folder_path = "data"
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            text += extract_text_from_pdf(pdf_path)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

print(text[:1000])

from datasets import Dataset

def split_text(text, chunk_size=300):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

chunks = split_text(text)

dataset = Dataset.from_dict({"text": chunks})

dataset = dataset.train_test_split(test_size=0.1)

from transformers import AutoTokenizer

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
    return tokenizer(
        example['text'],
        padding='max_length',
        truncation=True,
        max_length=128
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

tokenized_dataset = tokenized_dataset.map(
    lambda x: {
        "input_ids": x["input_ids"],
        "attention_mask": x["attention_mask"],
        "labels": x["input_ids"]
    },
    remove_columns=tokenized_dataset["train"].column_names
)

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./qlora-law",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="epoch",     #  FIXED
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

trainer.train()

# 


import math

last_eval = [x for x in trainer.state.log_history if "eval_loss" in x][-1]

print("Eval Loss:", last_eval["eval_loss"])
print("Perplexity:", math.exp(last_eval["eval_loss"]))

# **Interpretation**
#  Model Performance
# * Low eval loss (~1.34) → model learned well
# * Perplexity ~3.84 → very strong predictions
# 
#  Rule of thumb:
# 
# - < 10 ->  Good
# - < 5 -> Very good (your case)


model.save_pretrained("qlora-law-model")
tokenizer.save_pretrained("qlora-law-model")

# ================================
# Load YOUR saved model
# ================================
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

SAVE_DIR = "qlora-law-model"
model_name = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, SAVE_DIR)

tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
tokenizer.pad_token = tokenizer.eos_token


# ================================
# Your model pipeline
# ================================
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# ================================
# Evaluator model
# ================================
from transformers import pipeline

evaluator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

def evaluate_answer(question, generated_answer):
    prompt = f"""
Question: {question}

Answer: {generated_answer}

Give score (0-10) for correctness and explain.
"""

    result = evaluator(prompt, max_new_tokens=100)[0]["generated_text"]
    return result


# ================================
# Test
# ================================
question = "What is Indian Penal Code?"

generated = pipe(
    f"Question: {question}\nAnswer:",
    max_new_tokens=150
)[0]["generated_text"]

evaluation = evaluate_answer(question, generated)

print("\n=== GENERATED ===\n", generated)
print("\n=== EVALUATION ===\n", evaluation)

model = model.merge_and_unload()
model.save_pretrained("final-model")

# ================================
# Load Saved Model
# ================================
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

SAVE_DIR = "qlora-law-model"
model_name = "microsoft/phi-2"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, SAVE_DIR)

tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
tokenizer.pad_token = tokenizer.eos_token


# ================================
# Pipeline
# ================================
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)


# ================================
# Ask Multiple Questions
# ================================
questions = [
    "What is Indian Penal Code?",
    "What is bail?",
    "What is FIR?",
    "What are non-bailable offences?",
    "Explain section 420 IPC",
]

for q in questions:
    prompt = f"You are a legal expert.\nQuestion: {q}\nAnswer:"

    output = pipe(
        prompt,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    print("\n" + "="*50)
    print("QUESTION:", q)
    print("-"*50)
    print(output[0]["generated_text"])