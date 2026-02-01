import os
import sys
import math
import torch
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    pipeline,
    DataCollatorForLanguageModeling
)


class CustomTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


class CustomDataCollatorForLanguageModeling:
    def __init__(self, model, tokenizer, mask_probability=0.15, lexicon=None):
        self.model = model
        self.tokenizer = tokenizer
        self.mask_probability = mask_probability
        self.lexicon = lexicon

    def __call__(self, batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = input_ids.clone()

        lexicon_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.lexicon:
            input_ids, lexicon_mask = self._apply_lexicon_masking(input_ids)

        probability_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.mask_probability > 0:
            input_ids, probability_mask = self._apply_probability_masking(input_ids, lexicon_mask)

        labels[(~lexicon_mask) & (~probability_mask)] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def _apply_lexicon_masking(self, input_ids):
        lexicon_token_sequences = [self.tokenizer.encode(word.strip(), add_special_tokens=False) for word in self.lexicon]
        mask_token_id = self.tokenizer.mask_token_id
        lexicon_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for seq in lexicon_token_sequences:
            seq_len = len(seq)
            for i in range(len(input_ids)):
                for j in range(len(input_ids[i]) - seq_len + 1):
                    if input_ids[i][j:j + seq_len].tolist() == seq:
                        input_ids[i][j:j + seq_len] = torch.tensor([mask_token_id] * seq_len, dtype=input_ids.dtype)
                        lexicon_mask[i][j:j + seq_len] = True

        return input_ids, lexicon_mask

    def _apply_probability_masking(self, input_ids, lexicon_mask):
        total_tokens = input_ids.numel()
        max_masked_tokens = int(total_tokens * self.mask_probability)
        remaining_masked_tokens = max_masked_tokens - lexicon_mask.sum().item()

        if remaining_masked_tokens <= 0:
            return input_ids, torch.zeros_like(input_ids, dtype=torch.bool)

        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for special_token_id in self.tokenizer.all_special_ids:
            special_tokens_mask[input_ids == special_token_id] = True

        probability_mask = torch.rand(input_ids.shape) < (remaining_masked_tokens / total_tokens)
        probability_mask[input_ids == self.tokenizer.pad_token_id] = False
        probability_mask[lexicon_mask] = False
        input_ids[probability_mask] = self.tokenizer.mask_token_id

        return input_ids, probability_mask


def cargar_lexico(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lexicon = [line.strip() for line in f if line.strip()]
    return lexicon

def cargar_dataset(file_path):
    return pd.read_csv(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain adaptation")
    parser.add_argument('--train_file', type=str, required=True, help='Ruta al archivo CSV de entrenamiento')
    parser.add_argument('--lexicon_file', type=str, required=True, help='Ruta al archivo de léxico')
    args = parser.parse_args()

    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_mlm = AutoModelForMaskedLM.from_pretrained(model_name)

    df_train = cargar_dataset(args.train_file)
    lexico = cargar_lexico(args.lexicon_file)


    X = df_train["text"]

    train_dataset = CustomTextDataset(texts=X.tolist(), tokenizer=tokenizer)
    data_collator = CustomDataCollatorForLanguageModeling(model=model_mlm, tokenizer=tokenizer, lexicon=lexico)

    training_args_mlm = TrainingArguments(
        output_dir="./results_beto_adapted_mlm",
        overwrite_output_dir=True,
        num_train_epochs=4,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        save_strategy="no",
        report_to="none",
    )

    trainer_mlm = Trainer(
        model=model_mlm,
        args=training_args_mlm,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer_mlm.train()
    trainer_mlm.save_model("./models/beto-mlm-savemodel")
    model_mlm.save_pretrained("./models/beto-mlm-savepretrained")
