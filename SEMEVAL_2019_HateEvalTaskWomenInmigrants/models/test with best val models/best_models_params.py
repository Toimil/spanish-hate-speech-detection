###########################################
###########################################
###########################################
###########################################


models_dict = {
    ###########################################
    #semeval_val_lower_preprocess_without_url
    ###########################################

    './models/robertuito2epoch-superset-mlm-savemodel1': {
        'train_file': './semeval_train_lower_preprocess_without_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_url.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito2epoch-superset-mlm-savemodel',
        'learning_rate': 5.502325367492085e-06,
        'weight_decay': 0.0014713287026389913,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/semeval_test_lower_preprocess_without_url'
    },

    './models/robertuito-superset-mlm-savemodel1': {
        'train_file': './semeval_train_lower_preprocess_without_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_url.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito-superset-mlm-savemodel',
        'learning_rate': 1.9303427344279065e-05,
        'weight_decay': 0.0021864532786573087,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Robertuito_4epoch_superset/semeval_test_lower_preprocess_without_url'
    },

    'pysentimiento/robertuito-base-uncased1': {
        'train_file': './semeval_train_lower_preprocess_without_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_url.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 1.9266334139236565e-05,
        'weight_decay': 0.00035923919975377653,
        'num_train_epochs': 4,
        'batch_size': 16,
        'dir': './Robertuito_uncased/semeval_test_lower_preprocess_without_url'
    },
    ################
    #################

    'PlanTL-GOB-ES/roberta-base-bne1': {
        'train_file': './semeval_train_lower_preprocess_without_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_url.csv',
        'emb_size': 512,
        'tokenizer': 'PlanTL-GOB-ES/roberta-base-bne',
        'learning_rate': 1.0786405222092006e-05,
        'weight_decay': 8.583766433707602e-05,
        'num_train_epochs': 4,
        'batch_size': 16,
        'dir': './Roberta_bne/semeval_test_lower_preprocess_without_url'
    },

    'dccuchile/bert-base-spanish-wwm-uncased1': {
        'train_file': './semeval_train_lower_preprocess_without_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_url.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 1.9128765209791385e-05,
        'weight_decay': 0.0006952691084598837,
        'num_train_epochs': 6,
        'batch_size': 16,
        'dir': './Beto_uncased/semeval_test_lower_preprocess_without_url'
    },

    'bertin-project/bertin-roberta-base-spanish1': {
        'train_file': './semeval_train_lower_preprocess_without_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_url.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 1.2218122026654202e-05,
        'weight_decay': 0.002533569600382123,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Bertin/semeval_test_lower_preprocess_without_url'
    },
    


    ###########################################
    #semeval_val_lower
    ###########################################

    './models/robertuito2epoch-superset-mlm-savemodel2': {
        'train_file': './semeval_train_lower.csv',
        'val_file': './semeval_val_lower.csv',
        'test_file': './semeval_test_lower.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito2epoch-superset-mlm-savemodel',
        'learning_rate': 1.536615823221594e-05,
        'weight_decay': 0.0006329811352089739,
        'num_train_epochs': 4,
        'batch_size': 32,
        'dir': './Robertuito_2epoch_superset/semeval_test_lower'
    },

    ################
    #################
    './models/robertuito-superset-mlm-savemodel2': {
        'train_file': './semeval_train_lower.csv',
        'val_file': './semeval_val_lower.csv',
        'test_file': './semeval_test_lower.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito-superset-mlm-savemodel',
        'learning_rate': 7.694321484247116e-06,
        'weight_decay': 0.005127107101304225,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Robertuito_4epoch_superset/semeval_test_lower'
    },

    'pysentimiento/robertuito-base-uncased3': {
        'train_file': './semeval_train_lower.csv',
        'val_file': './semeval_val_lower.csv',
        'test_file': './semeval_test_lower.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 1.948641997608601e-05,
        'weight_decay':  0.0019486134378892688,
        'num_train_epochs': 6,
        'batch_size': 16,
        'dir': './Robertuito_uncased/semeval_test_lower'
    },

    'PlanTL-GOB-ES/roberta-base-bne3': {
        'train_file': './semeval_train_lower.csv',
        'val_file': './semeval_val_lower.csv',
        'test_file': './semeval_test_lower.csv',
        'emb_size': 512,
        'tokenizer': 'PlanTL-GOB-ES/roberta-base-bne',
        'learning_rate': 1.2301053348723977e-05,
        'weight_decay': 0.0063702446373470395,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Roberta_bne/semeval_test_lower'
    },

    'dccuchile/bert-base-spanish-wwm-uncased3': {
        'train_file': './semeval_train_lower.csv',
        'val_file': './semeval_val_lower.csv',
        'test_file': './semeval_test_lower.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 4.415501682117839e-06,
        'weight_decay': 0.00134394319748472,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Beto_uncased/semeval_test_lower'
    },
    
    ################
    #################

    'bertin-project/bertin-roberta-base-spanish3': {
        'train_file': './semeval_train_lower.csv',
        'val_file': './semeval_val_lower.csv',
        'test_file': './semeval_test_lower.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 7.869294771537981e-06,
        'weight_decay': 7.970257521999166e-05,
        'num_train_epochs': 3,
        'batch_size': 8,
        'dir': './Bertin/semeval_test_lower'
    },





    ###########################################
    #semeval_val_lower_preprocess_without_emojis_and_url
    ###########################################

    './models/robertuito2epoch-superset-mlm-savemodel4': {
        'train_file': './semeval_train_lower_preprocess_without_emojis_and_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito2epoch-superset-mlm-savemodel',
        'learning_rate': 1.0837062354583617e-05,
        'weight_decay': 0.007417520329154025,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/semeval_test_lower_preprocess_without_emojis_and_url'
    },

    './models/robertuito-superset-mlm-savemodel4': {
        'train_file': './semeval_train_lower_preprocess_without_emojis_and_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './models/robertuito-superset-mlm-savemodel',
        'learning_rate': 1.2506064859122329e-05,
        'weight_decay': 0.009711053219927043,
        'num_train_epochs': 5,
        'batch_size': 16,
        'dir': './Robertuito_4epoch_superset/semeval_test_lower_preprocess_without_emojis_and_url'
    },
    ################
    #################

    'pysentimiento/robertuito-base-uncased4': {
        'train_file': './semeval_train_lower_preprocess_without_emojis_and_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 1.1204401528305523e-05,
        'weight_decay':  0.0059889069216097456,
        'num_train_epochs': 6,
        'batch_size': 32,
        'dir': './Robertuito_uncased/semeval_test_lower_preprocess_without_emojis_and_url'
    },
        


    'PlanTL-GOB-ES/roberta-base-bne4': {
        'train_file': './semeval_train_lower_preprocess_without_emojis_and_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'PlanTL-GOB-ES/roberta-base-bne',
        'learning_rate': 1.431998829901923e-05,
        'weight_decay': 5.82218784068546e-05,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Roberta_bne/semeval_test_lower_preprocess_without_emojis_and_url'
    },

    'dccuchile/bert-base-spanish-wwm-uncased4': {
        'train_file': './semeval_train_lower_preprocess_without_emojis_and_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 5.823883858173515e-06,
        'weight_decay':  0.00044457312331044494,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Beto_uncased/semeval_test_lower_preprocess_without_emojis_and_url'
    },

    'bertin-project/bertin-roberta-base-spanish4': {
        'train_file': './semeval_train_lower_preprocess_without_emojis_and_url.csv',
        'val_file': './semeval_val_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './semeval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 1.976433274431061e-06,
        'weight_decay': 5.5975045767398875e-05,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Bertin/semeval_test_lower_preprocess_without_emojis_and_url'
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

DATASETS_DELIMITERS = {
    'SEMEVAL_2019': '	',
}
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
        ) | {"labels": examples["hate_speech"]}

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
        skiprows=1,
        header=None,
        names = ["id", "text", "hate_speech", "target_range", "aggressiveness"],
        delimiter=DATASETS_DELIMITERS['SEMEVAL_2019']
    )
    return df


if __name__ == "__main__":
    os.environ['TORCH_USE_CUDA_DSA'] = "1"
    random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)


    for model_name, config in models_dict.items():

        df_train = cargar_dataset(config['train_file'])
        df_val = cargar_dataset(config['val_file'])
        df_test = cargar_dataset(config['test_file'])

        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_val),
            "test": Dataset.from_pandas(df_test)
        })

        classifier = HateSpeechClassifier(config['tokenizer'], config, dataset_dict)
        classifier.train_and_predict()



   


