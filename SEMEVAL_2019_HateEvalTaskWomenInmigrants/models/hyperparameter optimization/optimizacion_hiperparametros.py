# Imports
import sys
import pandas as pd
import numpy as np
import optuna
import operator
import argparse
import torch
import random
import os
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split


# Constantes
DATASETS_DELIMITERS = {
    'SEMEVAL_2019': '	',
}
SEED_VALUE = 1

models_dict = {
    'pysentimiento/robertuito-base-uncased': {
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'dir': './Robertuito_uncased'
    },
    'dccuchile/bert-base-spanish-wwm-uncased': {
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'dir': './Beto_uncased'
    },
    'PlanTL-GOB-ES/roberta-base-bne': {
        'emb_size': 512,
        'tokenizer': 'PlanTL-GOB-ES/roberta-base-bne',
        'dir': './Roberta_bne'
    },
    'bertin-project/bertin-roberta-base-spanish': {
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'dir': './Bertin'
    },
}


# Argumentos
parser = argparse.ArgumentParser(description="Fine-tuning y evaluación de modelos para misoginia")
parser.add_argument('--train_file', type=str, required=True, help='Ruta al archivo CSV de train')
parser.add_argument('--val_file', type=str, required=True, help='Ruta al archivo CSV de validación')
args = parser.parse_args()
if not args.train_file or not args.val_file:
    parser.print_help()
    sys.exit(1)


# Reproducibilidad
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED_VALUE)


def cargar_dataset(ruta):
    df = pd.read_csv(
        ruta,
        skiprows=1,
        header=None,
        names = ["id", "text", "hate_speech", "target_range", "aggressiveness"],
        delimiter=DATASETS_DELIMITERS['SEMEVAL_2019']
    )
    return df


df_train = cargar_dataset(args.train_file)
df_val = cargar_dataset(args.val_file)


X_train = df_train["text"]
y_train = df_train["hate_speech"]
X_val = df_val["text"]
y_val = df_val["hate_speech"]

train_df = pd.DataFrame({"text": X_train, "hate_speech": y_train})
val_df = pd.DataFrame({"text": X_val, "hate_speech": y_val})
train_df.to_csv("semeval_train_split.csv", index=False)
val_df.to_csv("semeval_val_split.csv", index=False)


class EarlyStoppingOptuna:
    def __init__(self, early_stopping_rounds: int, direction: str = "maximize") -> None:
        self.early_stopping_rounds = early_stopping_rounds
        self._iter = 0
        self._score = -np.inf if direction == "maximize" else np.inf
        self._operator = operator.gt if direction == "maximize" else operator.lt

    def __call__(self, study: optuna.Study, trial: optuna.Trial):
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1
        if self._iter >= self.early_stopping_rounds:
            study.stop()


class CustomTrainer(Trainer):
    def log(self, logs, iterator_start_time=None):
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs, iterator_start_time)


class HateSpeechClassifier:
    def __init__(self, model_name, config, train_file_name):
        self.model_name = model_name
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], model_max_length=config['emb_size'])

        output_dir = os.path.join(self.config['dir'], f"{train_file_name}_output")
        os.makedirs(output_dir, exist_ok=True)
        self.config['dir'] = output_dir

    def preprocess_text(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            max_length=self.config['emb_size']
        ) | {"labels": examples["hate_speech"]}

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        print("\n--- Métricas en conjunto de evaluación ---")
        print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
        print(f"Precision: {precision_score(labels, preds, average='macro'):.4f}")
        print(f"Recall:    {recall_score(labels, preds, average='macro'):.4f}")
        print(f"F1-macro:  {f1_score(labels, preds, average='macro'):.4f}")
        print(f"F1-micro:  {f1_score(labels, preds, average='micro'):.4f}")
        print(f"F1-score:  {f1_score(labels, preds):.4f}")
        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='macro'),
            'recall': recall_score(labels, preds, average='macro'),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'f1_score': f1_score(labels, preds)
        }

    def objective(self, trial):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        model.config.id2label = {0: 'No hate_speech', 1: 'hate_speech'}
        model.config.label2id = {'No hate_speech': 0, 'hate_speech': 1}

        dataset = load_dataset('csv', data_files={'train': 'semeval_train_split.csv', 'validation': 'semeval_val_split.csv'})
        tokenized_data = dataset.map(self.preprocess_text, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trial_dir = os.path.join(self.config['dir'], str(trial.number))
        os.makedirs(trial_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=trial_dir,
            learning_rate=trial.suggest_loguniform('learning_rate', 7e-7, 2e-5),
            weight_decay=trial.suggest_loguniform('weight_decay', 4e-5, 0.01),
            num_train_epochs=trial.suggest_int('num_train_epochs', 3, 6),
            per_device_train_batch_size=trial.suggest_categorical('batch_size', [8, 16, 32]),
            per_device_eval_batch_size=trial.suggest_categorical('batch_size', [8, 16, 32]),
            eval_strategy ="epoch",
            save_strategy="no",
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            seed=SEED_VALUE
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print(f"Trial {trial.number} - Model: {self.model_name}")
        print(f"Learning Rate: {training_args.learning_rate}")
        print(f"Weight Decay: {training_args.weight_decay}")
        print(f"Num Train Epochs: {training_args.num_train_epochs}")
        print(f"Batch Size: {training_args.per_device_train_batch_size}")
        print(f"Output Dir: {training_args.output_dir}")

        trainer.train()
        result = trainer.evaluate()
        pd.DataFrame(result, index=[0]).to_csv(
            os.path.join(trial_dir, "evaluationResult.txt"), sep='\t', index=False
        )

        return result["eval_f1_macro"]

    def train(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=8, callbacks=[EarlyStoppingOptuna(4)])
        with open(os.path.join(self.config['dir'], "results.txt"), 'w') as f:
            f.write(f"Best F1 Score: {study.best_value}\nBest Params: {study.best_params}\nBest Trial: {study.best_trial}\n")

for model_name, config in models_dict.items():
    print(f"\nEntrenando modelo: {model_name}\n")
    print(f"Train file: {args.train_file}")
    train_file_name = os.path.splitext(os.path.basename(args.train_file))[0]
    trainer = HateSpeechClassifier(model_name, config, train_file_name)
    trainer.train()
