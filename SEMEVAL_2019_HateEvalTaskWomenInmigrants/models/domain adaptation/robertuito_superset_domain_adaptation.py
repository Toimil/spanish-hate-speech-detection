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

DATASETS_DELIMITERS = {
    'SEMEVAL_2019': '	',
}
SEED_VALUE = 1


class CustomTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
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

        texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, return_attention_mask=False, return_offsets_mapping=False)
        word_ids_batch = [self.tokenizer(text, return_offsets_mapping=True).word_ids() for text in texts]

        lexicon_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.lexicon:
            input_ids, lexicon_mask = self._apply_lexicon_masking(input_ids, word_ids_batch)

        probability_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.mask_probability > 0:
            input_ids, probability_mask = self._apply_probability_masking(input_ids, lexicon_mask, word_ids_batch)

        labels[(~lexicon_mask) & (~probability_mask)] = -100
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def _apply_lexicon_masking(self, input_ids, word_ids_batch):
        mask_token_id = self.tokenizer.mask_token_id
        lexicon_set = set([w.lower() for w in self.lexicon])
        lexicon_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i, (input_ids_seq, w_ids) in enumerate(zip(input_ids, word_ids_batch)):
            word_to_token_indices = {}
            for idx, word_id in enumerate(w_ids):
                if word_id is None:
                    continue
                word_to_token_indices.setdefault(word_id, []).append(idx)

            for word_id, token_indices in word_to_token_indices.items():
                word_tokens = input_ids_seq[token_indices]
                decoded_word = self.tokenizer.decode(word_tokens, skip_special_tokens=True).strip().lower()

                if decoded_word in lexicon_set:
                    for idx in token_indices:
                        input_ids[i, idx] = mask_token_id
                        lexicon_mask[i, idx] = True

        return input_ids, lexicon_mask

    def _apply_probability_masking(self, input_ids, lexicon_mask, word_ids_batch):
        mask_token_id = self.tokenizer.mask_token_id
        probability_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for i, (input_ids_seq, w_ids, lex_mask) in enumerate(zip(input_ids, word_ids_batch, lexicon_mask)):
            word_to_token_indices = {}
            for idx, word_id in enumerate(w_ids):
                if word_id is None:
                    continue
                word_to_token_indices.setdefault(word_id, []).append(idx)

            for word_id, token_indices in word_to_token_indices.items():
                if any(lex_mask[token_indices]):
                    continue

                if torch.rand(1).item() < self.mask_probability:
                    for idx in token_indices:
                        if input_ids_seq[idx] not in self.tokenizer.all_special_ids:
                            input_ids[i, idx] = mask_token_id
                            probability_mask[i, idx] = True

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

    model_name = "pysentimiento/robertuito-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_mlm = AutoModelForMaskedLM.from_pretrained(model_name)

    # Cargar datasets
    df_train = cargar_dataset(args.train_file)
    lexico_misoginia = cargar_lexico(args.lexicon_file)

    X = df_train["text"]

    train_dataset = CustomTextDataset(texts=X.tolist(), tokenizer=tokenizer)
    data_collator = CustomDataCollatorForLanguageModeling(model=model_mlm, tokenizer=tokenizer, lexicon=lexico_misoginia)

    training_args_mlm = TrainingArguments(
        output_dir="./results_superset_robertuito_mlm",
        overwrite_output_dir=True,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        save_strategy="no",
        report_to="none",
        seed=SEED_VALUE
    )

    trainer_mlm = Trainer(
        model=model_mlm,
        args=training_args_mlm,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer_mlm.train()
    trainer_mlm.save_model("./models/robertuito-superset-mlm-savemodel")
    model_mlm.save_pretrained("./models/robertuito-superset-mlm-savepretrained")

