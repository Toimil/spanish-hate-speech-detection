###########################################
###########################################
###########################################
###########################################


models_dict = {
    ###########################################
    #ibereval_train_lower_preprocess_without_emojis_and_url
    ###########################################


    'bertin-project/bertin-roberta-base-spanish1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 1.1311915743815852e-05,
        'weight_decay': 7.19812038936289e-05,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Bertin/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    'dccuchile/bert-base-spanish-wwm-uncased1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 9.588237690667472e-06,
        'weight_decay': 0.0030169732085897835,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Beto_uncased/ibereval_train_lower_preprocess_without_emojis_and_url'
    },


    'pysentimiento/robertuito-base-uncased1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 1.5130168937030695e-05,
        'weight_decay': 0.00012307483097166058,
        'num_train_epochs': 6,
        'batch_size': 32,
        'dir': './Robertuito_uncased/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    './models/beto-mlm-savemodel1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': './model-beto-mlm-adaptado-misoginia-4epoch',
        'learning_rate': 1.1644940883516713e-05,
        'weight_decay': 0.0002648899625163922,
        'num_train_epochs': 3,
        'batch_size': 16,
        'dir': './Beto_4epoch_superset/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    './models/robertuito2epoch-superset-mlm-savemodel1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-2epoch',
        'learning_rate': 1.816289058177142e-05,
        'weight_decay':  0.00038009825372032626,
        'num_train_epochs': 3,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

    './models/robertuito-superset-mlm-savemodel1': {
        'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-4epoch',
        'learning_rate': 1.5123286783562446e-05,
        'weight_decay': 0.0008073729589612988,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Robertuito_4epoch_superset/ibereval_train_lower_preprocess_without_emojis_and_url'
    },

     'PlanTL-GOB-ES/roberta-base-bne1': {
         'train_file': './ibereval_train_lower_preprocess_without_emojis_and_url.csv',
         'test_file': './ibereval_test_lower_preprocess_without_emojis_and_url.csv',
         'emb_size': 512,
         'tokenizer': 'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy',
         'learning_rate': 8.756444914821502e-06,
         'weight_decay': 0.003618080144832499,
         'num_train_epochs': 5,
         'batch_size': 32,
         'dir': './Roberta_bne/ibereval_train_lower_preprocess_without_emojis_and_url'
     },
    


    ###########################################
    #ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url
    ###########################################

    'bertin-project/bertin-roberta-base-spanish3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate': 9.60775267925975e-06,
        'weight_decay': 5.2925167720881785e-05,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Bertin/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },
    
    'dccuchile/bert-base-spanish-wwm-uncased3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate':  1.1117016430997405e-05,
        'weight_decay':  0.00014983824173514133,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Beto_uncased/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    'pysentimiento/robertuito-base-uncased3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 1.5206411136497554e-05,
        'weight_decay':  0.008094780823686483,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Robertuito_uncased/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    './models/beto-mlm-savemodel3': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 512,
        'tokenizer': './model-beto-mlm-adaptado-misoginia-4epoch',
        'learning_rate': 1.7253838023061347e-05,
        'weight_decay': 0.00159773804737677,
        'num_train_epochs': 5,
        'batch_size': 16,
        'dir': './Beto_4epoch_superset/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    './models/robertuito2epoch-superset-mlm-savemodel2': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-2epoch',
        'learning_rate': 1.8778497011004125e-05,
        'weight_decay':  0.00011907016799923736,
        'num_train_epochs': 4,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },

    './models/robertuito-superset-mlm-savemodel2': {
        'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-4epoch',
        'learning_rate': 1.6705670521215263e-05,
        'weight_decay': 7.260410412123703e-05,
        'num_train_epochs': 5,
        'batch_size': 32,
        'dir': './Robertuito_4epoch_superset/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
    },
     'PlanTL-GOB-ES/roberta-base-bne3': {
         'train_file': './ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
         'test_file': './ibereval_test_lower_preprocess_without_hashtags_and_emojis_and_url.csv',
         'emb_size': 512,
         'tokenizer': 'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy',
         'learning_rate': 8.526488404343225e-06,
         'weight_decay': 0.002129883009207547,
         'num_train_epochs': 3,
         'batch_size': 8,
         'dir': './Roberta_bne/ibereval_train_lower_preprocess_without_hashtags_and_emojis_and_url'
     },




    ###########################################
    #ibereval_train_lower_preprocess_default
    ###########################################


    'bertin-project/bertin-roberta-base-spanish4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': 'bertin-project/bertin-roberta-base-spanish',
        'learning_rate':  4.636435322035031e-06,
        'weight_decay': 0.00401752318441989,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Bertin/ibereval_train_lower_preprocess_default'
    },
    
    'dccuchile/bert-base-spanish-wwm-uncased4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': 'dccuchile/bert-base-spanish-wwm-uncased',
        'learning_rate': 1.029469543364061e-05,
        'weight_decay':  0.0024346483100308514,
        'num_train_epochs': 6,
        'batch_size': 32,
        'dir': './Beto_uncased/ibereval_train_lower_preprocess_default'
    },

    'pysentimiento/robertuito-base-uncased4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 128,
        'tokenizer': 'pysentimiento/robertuito-base-uncased',
        'learning_rate': 8.656742715195223e-06,
        'weight_decay': 0.0037733848495952976,
        'num_train_epochs': 5,
        'batch_size': 8,
        'dir': './Robertuito_uncased/ibereval_train_lower_preprocess_default'
    },
    
    './models/beto-mlm-savemodel4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': './model-beto-mlm-adaptado-misoginia-4epoch',
        'learning_rate':  1.1960723115975357e-05,
        'weight_decay': 4.769514983226764e-05,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Beto_4epoch_superset/ibereval_train_lower_preprocess_default'
    },

    './models/robertuito2epoch-superset-mlm-savemodel4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-2epoch',
        'learning_rate':  8.240867404336875e-06,
        'weight_decay': 0.0002012862981135354,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Robertuito_2epoch_superset/ibereval_train_lower_preprocess_default'
    },

    './models/robertuito-superset-mlm-savemodel4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 128,
        'tokenizer': './model-robertuito-mlm-adaptado-misoginia-4epoch',
        'learning_rate': 1.7285092285326795e-05,
        'weight_decay':  8.828925742569351e-05,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Robertuito_4epoch_superset/ibereval_train_lower_preprocess_default'
    },

    'PlanTL-GOB-ES/roberta-base-bne4': {
        'train_file': './ibereval_train_lower_preprocess_default.csv',
        'test_file': './ibereval_test_lower_preprocess_default.csv',
        'emb_size': 512,
        'tokenizer': 'PeterPanecillo/PlanTL-GOB-ES-roberta-base-bne-copy',
        'learning_rate':  1.5227932285555862e-05,
        'weight_decay':  0.00038718259929214566,
        'num_train_epochs': 6,
        'batch_size': 8,
        'dir': './Roberta_bne/ibereval_train_lower_preprocess_default'
    },

}



###########################################
###########################################
###########################################
###########################################







import gc
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
    def __init__(self, model_name, config, dataset_dict, y_col):
        self.model_name = model_name
        self.config = config
        self.dataset = dataset_dict
        self.tokenizer = AutoTokenizer.from_pretrained(
            config['tokenizer'],
            model_max_length=config['emb_size']
        )
        self.y_col = y_col
        self.labels = ['0', 'discredit', 'dominance', 'sexual_harassment', 'stereotype', 'derailing']
        self.label2id = {'0': 0, 'discredit': 1, 'dominance': 2, 'sexual_harassment': 3, 'stereotype': 4, 'derailing': 5}
        self.id2label = {0: '0', 1: 'discredit', 2: 'dominance', 3: 'sexual_harassment', 4: 'stereotype', 5: 'derailing'}


    def preprocess_text(self, examples):
        inputs = [str(text) for text in examples["text"]]
        
        tokenized = self.tokenizer(
            inputs,
            truncation=True,
            padding=False,
            max_length=self.config["emb_size"]
        )
        
        tokenized["labels"] = [int(self.label2id[l]) for l in examples["misogyny_category"]]
        
        return tokenized

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
        print(f"F1-score:  {f1_score(labels, preds, average='weighted'):.4f}")
        print("\nReporte completo:")
        print(classification_report(labels, preds))

        return {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='macro'),
            'recall': recall_score(labels, preds, average='macro'),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_micro': f1_score(labels, preds, average='micro'),
            'f1_score': f1_score(labels, preds, average='weighted')
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
        num_labels = len(self.labels)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels)
        
        model.config.id2label = self.id2label
        model.config.label2id = self.label2id

        tokenized_data = self.dataset.map(
            self.preprocess_text, 
            batched=True,
            remove_columns=self.dataset["train"].column_names 
        )
        tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
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
        

         # ======= LIMPIEZA COMPLETA DE MEMORIA POR TRIAL =======
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            del model
            del trainer

            with torch.no_grad():
                torch.cuda.empty_cache()
                gc.collect()

            print(f"[INFO] Memoria liberada correctamente")

        except Exception as e:
            print(f"[WARNING] Error liberando memoria : {e}")
        # ======================================================


def cargar_dataset(ruta):
    df = pd.read_csv(
        ruta,
        delimiter=DATASETS_DELIMITERS['IBEREVAL_MISOGYNY_2018']
    )
    return df


if __name__ == "__main__":
    random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED_VALUE)


    for model_name, config in models_dict.items():

        df_train_full = cargar_dataset(config['train_file'])
        X_train, X_val, y_train, y_val = train_test_split(
            df_train_full["text"],
            df_train_full["misogyny_category"],
            test_size=0.25,
            random_state=SEED_VALUE,
            stratify=df_train_full["misogyny_category"]
        )

        df_train = pd.DataFrame({"text": X_train, "misogyny_category": y_train}).reset_index(drop=True)
        df_val = pd.DataFrame({"text": X_val, "misogyny_category": y_val}).reset_index(drop=True)

        df_test_full = cargar_dataset(config['test_file'])
        df_test = df_test_full[["text", "misogyny_category"]].reset_index(drop=True)

        dataset_dict = DatasetDict({
            "train": Dataset.from_pandas(df_train),
            "validation": Dataset.from_pandas(df_val),
            "test": Dataset.from_pandas(df_test)
        })

        classifier = HateSpeechClassifier(config['tokenizer'], config, dataset_dict, y_col="misogyny_category")
        classifier.train_and_predict()



   


