#!/usr/bin/env python
# coding: utf-8

# # German to English Translator Using Transformers

# ## Importing NLP libraries

# In[1]:


import torch
from torch.utils.data import Dataset
import joblib
from rich import print
import project_evaluate
import json
CUDA_LAUNCH_BLOCKING=1
import tqdm
import math
import nltk
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import torch
from transformers import  get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from nltk.translate.bleu_score import corpus_bleu


# In[2]:


# pip install --upgrade rich joblib transformers nltk evaluate torch 


# ## Preparing Dataset for Training The Model

# In[3]:


train_data = project_evaluate.read_file('./train.labeled')
val_data = project_evaluate.read_file('./val.labeled')

train_english_sentences = train_data[0]  # a list of Trainng English sentences
train_german_sentences =train_data[1]  # a list of Trainng German sentences

val_english_sentences = val_data[0]  # a list of Validation English sentences
val_german_sentences =val_data[1]  # a list of Validation German sentences


# ## Defining Training Parameters

# In[4]:


# local_model_filename = 't5_german_english_model_final.pt'
load_this_model = True   #  False / True
LR = 2e-5
TRAINING_EPOCHS = 100
generation_max_len = 250
# model_str_for_loading_pretrained = 't5-base'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## Loading Locally Saved Model and Tokenizer Saved

# In[5]:


# tokenizer = AutoTokenizer.from_pretrained(model_str_for_loading_pretrained,model_max_length=250)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_str_for_loading_pretrained)

tokenizer = joblib.load('my_tokenizer')
model = joblib.load('my_model')


# ## Defining Training Classes and Functions

# ### Defining Dataset Class

# In[6]:


class GermanEnglishDataset(Dataset):
    def __init__(self, german_texts, english_texts,  max_len):
        self.german_texts = german_texts
        self.english_texts = english_texts
        self.max_len = max_len

    def __len__(self):
        return len(self.german_texts)

    def __getitem__(self, index):
        german_text = str(self.german_texts[index])
        english_text = str(self.english_texts[index])

        input_ids = tokenizer.encode(
            german_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        target_ids = tokenizer.encode(
            english_text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": input_ids.flatten(),
            "attention_mask": input_ids != 0,
            "target_ids": target_ids.flatten(),
            "target_attention_mask": target_ids != 0
        }


# ### Defining Scoring Functions

# In[7]:


def predict_english_sentences(german_sentences, model,  device):
    english_sentences = []
    for german_sentence in tqdm.tqdm(german_sentences):
        input_ids = tokenizer.encode(
            german_sentence,
            add_special_tokens=True,
            max_length=generation_max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        outputs = model.to(device).generate(input_ids=input_ids, max_length=generation_max_len, num_beams=4, early_stopping=True)
        english_sentence = tokenizer.decode(outputs[0], skip_special_tokens=True)
        english_sentences.append(english_sentence)
    return english_sentences

def calculate_bleu_score(model,device):
    actual_english_sentences = val_english_sentences[0:5]
    # Predict English sentences from a list of German sentences
    german_sentences = val_german_sentences[0:5]
    english_sentences_pred = predict_english_sentences(german_sentences, model,  device)
    print(f'{actual_english_sentences=}')
    print(f'{english_sentences_pred=}')
    
    bleu_score = project_evaluate.compute_metrics(english_sentences_pred,actual_english_sentences)
    # project_evaluate.compute_metrics(trnqaslated_list,test_data[1])
    # bleu_score = corpus_bleu(actual_sentences, predicted_sentences)
    print(f'BLEU score: {bleu_score:.2f}')
    return bleu_score


# ### Defining Training Function

# #### Training Using Custom Designed `bleu_loss` Function

# In[8]:


# # ## Training with bleu_loss

# import numpy as np
# import torch
# import nltk

# # Set up data loaders
# train_dataset = GermanEnglishDataset(train_german_sentences,train_english_sentences,  max_len=generation_max_len)
# val_dataset = GermanEnglishDataset(val_german_sentences,val_english_sentences,  max_len=generation_max_len)

# train_dataloader = DataLoader(train_dataset, batch_size=45, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=45, shuffle=False)


# model = joblib.load('my_model_bleu_loss')

# # Set up optimizer and scheduler
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
# num_train_steps = int(len(train_dataset) / 16 * 10)
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_train_steps
# )

# import sacrebleu

# def bleu_loss(target_ids, logits):
#     # Convert target_ids to text
#     references = ["".join([str(token.item()) for token in sentence if token != 0]) for sentence in target_ids]
    
#     # Convert logits to text
#     predictions = ["".join([str(torch.argmax(token).item()) for token in sentence if torch.argmax(token).item() != 0]) for sentence in logits]
    
#     # Calculate sacrebleu score
#     bleu_score = sacrebleu.corpus_bleu(predictions, [references])
    
#     # Return negative bleu_score as the loss
#     return -torch.tensor(bleu_score.score, dtype=torch.float32, requires_grad=True)

# for epoch in range(TRAINING_EPOCHS):
#     model.train()
#     total_loss = 0
#     for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
#         batch = {k: v.to(device) for k, v in batch.items()}
#         optimizer.zero_grad()
#         outputs = model(
#             input_ids=batch["input_ids"],
#             attention_mask=batch["attention_mask"],
#             labels=batch["target_ids"],
#             decoder_attention_mask=batch["target_attention_mask"],
#             return_dict=True
#         )
#         loss = bleu_loss(batch["target_ids"], outputs.logits)
#         total_loss += loss.item()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         scheduler.step()
        
# #         joblib.dump(model,'my_model')
# #         bleu_score = calculate_bleu_score(model, device)
        
#     avg_train_loss = total_loss / len(train_dataloader)

#     # Evaluate on validation set
#     model.eval()
#     total_val_loss = 0
#     with torch.no_grad():
#         for step, batch in enumerate(val_dataloader):
#             batch = {k: v.to(device) for k, v in batch.items()}
#             outputs = model(
#                 input_ids=batch["input_ids"],
#                 attention_mask=batch["attention_mask"],
#                 labels=batch["target_ids"],
#                 decoder_attention_mask=batch["target_attention_mask"],
#                 return_dict=True
#             )
#             loss = bleu_loss(batch["target_ids"], outputs.logits)
#             total_val_loss += loss.item()
#         avg_val_loss = total_val_loss / len(val_dataloader)
        
#     joblib.dump(model,'my_model_bleu_loss')
#     print(f"Epoch {epoch + 1}: Train loss = {avg_train_loss:.3f}, Val loss = {avg_val_loss:.3f}")


# In[9]:


# joblib.dump(model,'my_model_bleu_loss')


# In[10]:


# import nltk
# nltk.download('punkt')


# #### Calculating Using Regular Loss Function

# In[11]:


def train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs):
    global LR
    last_valid_loss = 9999
    last_training_loss = 9999

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["target_ids"],
                decoder_attention_mask=batch["target_attention_mask"],
                return_dict=True
            )
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)

        # Evaluate on validation set
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["target_ids"],
                    decoder_attention_mask=batch["target_attention_mask"],
                    return_dict=True
                )
                loss = outputs.loss
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_dataloader)

        # Calculate BLEU score
        bleu_score = calculate_bleu_score(model, device)

        # Print metrics
        print(f'{epoch=}')
        print(f'{bleu_score=}')
        print(f'{avg_train_loss=}')
        print(f'{avg_val_loss=}')

        # torch.save(model.state_dict(), local_model_filename)
        if(avg_val_loss<last_valid_loss):
            joblib.dump(model,'my_model')

        if(last_training_loss<avg_train_loss):
            LR = LR*0.9
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR,betas=(0.9, 0.98), eps=1e-5)
            print(f'{LR=}')
            
        if(last_training_loss==avg_train_loss):
            LR = LR*1.1
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR,betas=(0.9, 0.98), eps=1e-5)
            print(f'{LR=}')
        last_valid_loss = avg_val_loss
        last_training_loss = avg_train_loss


# ## Running Training Process

# In[12]:


# Set up data loaders
train_dataset = GermanEnglishDataset(train_german_sentences,train_english_sentences,  max_len=generation_max_len)
val_dataset = GermanEnglishDataset(val_german_sentences,val_english_sentences,  max_len=generation_max_len)

train_dataloader = DataLoader(train_dataset, batch_size=70, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=70, shuffle=False)

# Set up optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
num_train_steps = int(len(train_dataset) / 16 * 10)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)

# Train and evaluate the model
model.to(device)
train(model, train_dataloader, val_dataloader, optimizer, scheduler, device, epochs=TRAINING_EPOCHS)


# ## Calculating BLEU Score

# In[ ]:


pred_test_eng = predict_english_sentences(val_data[1][0:5], model,  device)
actual_test_eng = val_data[0][0:5]
bleu_score = project_evaluate.compute_metrics(pred_test_eng,actual_test_eng)
print(f'{bleu_score=}')


# In[ ]:


# joblib.dump(model,'my_model')


# In[ ]:




