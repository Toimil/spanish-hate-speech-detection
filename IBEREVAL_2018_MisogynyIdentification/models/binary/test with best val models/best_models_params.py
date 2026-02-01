###########################################
###########################################
###########################################
###########################################


models_dict = {
    ###########################################
    #ibereval_train_lower_preprocess_without_emojis_and_url
    ###########################################

    './models/robertuito2epoch-superset-mlm-savemodel1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito2epoch-superset-mlm-savemodel',
        'learning_rate': 8.22768837224868e-06,
        'weight_decay': 0.009354252751444993,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    './models/robertuito-superset-mlm-savemodel1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito-superset-mlm-savemodel',
        'learning_rate': 1.1068369690281625e-05,
        'weight_decay': 0.0006165771379145076,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Robertuito_4epoch_superset/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    './models/beto-mlm-savemodel1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': './models/beto-mlm-savemodel',
        'learning_rate': 1.9158924000405467e-05,
        'weight_decay': 0.00029378207818945694,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Beto_4epoch_superset/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    'pysentimiento/robertuito-base-uncased1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 7.042948154220254e-06,
        'weight_decay': 0.0030611647369572096,
        'num_train_epochs': 6,
        'batch_size': 16,
        'dir': './Robertuito_uncased/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    'PlanTL-GOB-ES/roberta-base-bne1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'PlanTL-GOB-ES/roberta-base-bne',
        'learning_rate': 1.5181881396874114e-05,
        'weight_decay': 0.00329019229039584,
        'num_train_epochs': 3,
        'batch_size': 16,
        'dir': './Roberta_bne/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    'dccuchile/bert-base-spanish-wwm-uncased1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 5.491164732515198e-06,
        'weight_decay': 0.005453872537248584,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Beto_uncased/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    'bertin-project/bertin-roberta-base-spanish1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 1.4566780133239426e-05,
        'weight_decay': 0.0015536415053288158,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Bertin/ibereval_train_lower_preprocess_without_emojis_and_url'
    },
    


    ###########################################
    #ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url
    ###########################################

    './models/robertuito2epoch-superset-mlm-savemodel2': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito2epoch-superset-mlm-savemodel',
        'learning_rate': 9.824459722658509e-06,
        'weight_decay': 0.00789333111874204,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    './models/robertuito-superset-mlm-savemodel2': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito-superset-mlm-savemodel',
        'learning_rate': 1.3426280538256742e-05,
        'weight_decay': 0.0004017550379722498,
        'num_train_epochs': 3,
        'batch_size': 16,
        'dir': './Robertuito_4epoch_superset/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    './models/beto-mlm-savemodel3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': './models/beto-mlm-savemodel',
        'learning_rate': 5.579865834931962e-06,
        'weight_decay': 6.335884359108968e-05,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Beto_4epoch_superset/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    'pysentimiento/robertuito-base-uncased3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 1.349623851035428e-05,
        'weight_decay':  0.00047408162907921163,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Robertuito_uncased/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    'PlanTL-GOB-ES/roberta-base-bne3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'PlanTL-GOB-ES/roberta-base-bne',
        'learning_rate': 8.330956613798717e-06,
        'weight_decay': 0.00036815583459295134,
        'num_train_epochs': 5,
        'batch_size': 16,
        'dir': './Roberta_bne/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    'dccuchile/bert-base-spanish-wwm-uncased3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 4.320266859419957e-06,
        'weight_decay': 0.0005021011164833326,
        'num_train_epochs': 6,
        'batch_size': 32,
        'dir': './Beto_uncased/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    'bertin-project/bertin-roberta-base-spanish3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 1.1556548673813666e-05,
        'weight_decay': 9.924014586725395e-05,
        'num_train_epochs': 3,
        'batch_size': 32,
        'dir': './Bertin/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },





    ###########################################
    #ibereval_train_lower_preprocess_default
    ###########################################

    './models/robertuito2epoch-superset-mlm-savemodel4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito2epoch-superset-mlm-savemodel',
        'learning_rate': 1.5157057561070255e-05,
        'weight_decay': 8.948493865766729e-05,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/ibereval_train_lower_preprocess_default'
    },

    './models/robertuito-superset-mlm-savemodel4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito-superset-mlm-savemodel',
        'learning_rate': 1.2911236544858202e-05,
        'weight_decay': 0.0022920652352782372,
        'num_train_epochs': 6,
        'batch_size': 16,
        'dir': './Robertuito_4epoch_superset/ibereval_train_lower_preprocess_default'
    },

    './models/beto-mlm-savemodel4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': './models/beto-mlm-savemodel',
        'learning_rate': 1.5787697895729105e-05,
        'weight_decay': 0.0002917779076196227,
        'num_train_epochs': 4,
        'batch_size': 32,
        'dir': './Beto_4epoch_superset/ibereval_train_lower_preprocess_default'
    },

    'pysentimiento/robertuito-base-uncased4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 1.886771365949286e-05,
        'weight_decay':  0.003325494903725337,
        'num_train_epochs': 5,
        'batch_size': 32,
        'dir': './Robertuito_uncased/ibereval_train_lower_preprocess_default'
    },

    'PlanTL-GOB-ES/roberta-base-bne4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': 'PlanTL-GOB-ES/roberta-base-bne',
        'learning_rate': 1.2221717338521869e-05,
        'weight_decay': 0.0002449513641283539,
        'num_train_epochs': 6,
        'batch_size': 16,
        'dir': './Roberta_bne/ibereval_train_lower_preprocess_default'
    },

    'dccuchile/bert-base-spanish-wwm-uncased4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 6.0481242514427294e-06,
        'weight_decay':  0.0004368919256502007,
        'num_train_epochs': 5,
        'batch_size': 16,
        'dir': './Beto_uncased/ibereval_train_lower_preprocess_default'
    },

    'bertin-project/bertin-roberta-base-spanish4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 1.7601146286831413e-05,
        'weight_decay': 0.00290730572450408,
        'num_train_epochs': 4,
        'batch_size': 32,
        'dir': './Bertin/ibereval_train_lower_preprocess_default'
    },
}



###########################################
###########################################
###########################################
###########################################







import os
import argparse
import pandas as pd
import numpy as np
import json
import torch
import random
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

DATASETS_DELIMITERS = {'IBEREVAL_MISOGYNY_2018': ','}
SEED_VALUE = 1
os.environ["WANDB_DISABLED"] = "true"

class CustomTrainer(Trainer):
    def log(self, logs, iterator_start_time=None):
        logs["learning_rate"] = self._get_learning_rate()
        super().log(logs, iterator_start_time)

class HateSpeechClassifier:
    def __init__(self, model_name, config, dataset_dict):
        self.model_name = model_name
        self.config = config
        self.dataset = dataset_dict
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['tokenizer'],
            model_max_length=config['emb_size']
        )

    def preprocess_text(self, examples):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding='max_length',
            max_length=self.config['emb_size']
        ) | {"labels": examples["misogynous"]}

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)

        print("\n--- Métricas -------")
        print("Model:", self.model_name)
        print("\n--- Métricas -------")
        print(f"Accuracy:  {accuracy_score(labels, preds):.4f}")
        print(f"Precision: {precision_score(labels, preds, average='macro'):.4f}")
        print(f"Recall:    {recall_score(labels, preds, average='macro'):.4f}")
        print(f"F1-macro:  {f1_score(labels, preds, average='macro'):.4f}")
        print(f"F1-micro:  {f1_score(labels, preds, average='micro'):.4f}")
        print(f"F1-score:  {f1_score(labels, preds):.4f}")
        print("\nReporte completo:")
        print(classification_report(labels, preds))

        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='macro'),
            'recall': recall_score(labels, preds, average='macro'),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'f1_score': f1_score(labels, preds)
        }

    def predict_and_save(self, trainer, dataset, filename, conjunto="test"):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        predictions = trainer.predict(dataset)
        preds = np.argmax(predictions.predictions, axis=1)

        df = pd.DataFrame({'predictions': preds})
        df.to_csv(filename, index=False)
        print(f" Predicciones guardadas en {filename}")

        metrics = self.compute_metrics((predictions.predictions, dataset["labels"]))
        print(f" Métricas calculadas para {conjunto}")

        metrics_filename = filename.replace(".csv", "_metrics.json")
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f" Métricas guardadas en {metrics_filename}")

        return preds, dataset["labels"]

    def train_and_predict(self):
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        model.config.id2label = {0: 'No misogynous', 1: 'Misogynous'}
        model.config.label2id = {'No misogynous': 0, 'Misogynous': 1}

        tokenized_data = self.dataset.map(self.preprocess_text, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
            output_dir=self.config['dir'],
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            num_train_epochs= self.config['num_train_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
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
            compute_metrics=self.compute_metrics
        )

        trainer.train()

        # Validacion
        self.predict_and_save(trainer, tokenized_data['validation'],
                      os.path.join(self.config['dir'], "predictions_validation.csv"),
                      conjunto="validacion")
        # Test
        self.predict_and_save(trainer, tokenized_data['test'],
                      os.path.join(self.config['dir'], "predictions_test.csv"),
                      conjunto="test")


def cargar_dataset(ruta):
    df = pd.read_csv(
        ruta,
        delimiter=DATASETS_DELIMITERS['IBEREVAL_MISOGYNY_2018']
    )
    return df


if __name__ == "__main__":
    # Fijar semillas
    random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)


    for model_name, config in models_dict.items():

        df_train_full = cargar_dataset(config['train_file'])
        X_train, X_val, y_train, y_val = train_test_split(
            df_train_full["text"],
            df_train_full["misogynous"],
            test_size=0.25,
            random_state=SEED_VALUE,
            stratify=df_train_full["misogynous"]
        )

        df_train = pd.DataFrame({"text": X_train, "misogynous": y_train})
        df_val = pd.DataFrame({"text": X_val, "misogynous": y_val})

        df_test = cargar_dataset(config['test_file'])

        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_val),
            "test": Dataset.from_pandas(df_test)
        })

        classifier = HateSpeechClassifier(config['tokenizer'], config, dataset_dict)
        classifier.train_and_predict()



   


